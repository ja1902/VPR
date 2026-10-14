"""
Query the FAISS index with a video to find the closest matching location.
Extracts frames at a configurable FPS, computes per-frame SALAD descriptors,
queries each frame individually, then uses geographic clustering of per-frame
matches to determine if the query is in-range and resolve the final location.

Usage:
    python 07_query_video.py <video_path>
    python 07_query_video.py <video_path> --rerank
    python 07_query_video.py <video_path> --fps 3 --max_frames 10 --rerank
    python 07_query_video.py <video_path> --crop 10,25 --rerank   # dashcam: trim 10% top, 25% bottom
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from . import config
from .model_utils import load_salad_model, get_transform, get_device
from .search import ReferenceDB
from .query_image import (weighted_centroid, print_results,
                          RERANK_MARGIN_THRESHOLD, CENTROID_TOP_M)

CLUSTER_RADIUS_M = 100
MIN_INLIER_RATIO = 0.4
MIN_INLIER_COUNT = 3


def clean_frame(pil_img, crop_pct=None):
    arr = np.array(pil_img)
    h, w = arr.shape[:2]

    # Auto-detect letterboxing (near-black rows/columns)
    row_means = arr.mean(axis=(1, 2))
    col_means = arr.mean(axis=(0, 2))
    threshold = 15

    top = 0
    while top < h and row_means[top] < threshold:
        top += 1
    bottom = h
    while bottom > top and row_means[bottom - 1] < threshold:
        bottom -= 1
    left = 0
    while left < w and col_means[left] < threshold:
        left += 1
    right = w
    while right > left and col_means[right - 1] < threshold:
        right -= 1

    arr = arr[top:bottom, left:right]

    if crop_pct is not None:
        ch, cw = arr.shape[:2]
        crop_top = int(ch * crop_pct[0] / 100)
        crop_bottom = int(ch * crop_pct[1] / 100)
        arr = arr[crop_top:ch - crop_bottom]

    return Image.fromarray(arr)


def extract_frames(video_path, sample_fps=config.VIDEO_SAMPLE_FPS,
                   max_frames=config.VIDEO_MAX_FRAMES, crop_pct=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or total_frames <= 0:
        raise ValueError(f"Invalid video metadata: fps={video_fps}, frames={total_frames}")

    frame_interval = max(1, int(video_fps / sample_fps))
    sample_indices = list(range(0, total_frames, frame_interval))[:max_frames]

    frames = []
    for target_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb)
        frames.append(clean_frame(pil_frame, crop_pct))

    cap.release()

    duration = total_frames / video_fps
    print(f"Video: {video_fps:.1f} fps, {total_frames} frames, {duration:.1f}s")
    crop_info = f", crop={crop_pct[0]}%/{crop_pct[1]}%" if crop_pct else ""
    print(f"Extracted {len(frames)} frames (sample fps={sample_fps}, interval={frame_interval}{crop_info})")

    return frames


def seqgem_pool(descriptors, p=3.0, eps=1e-6):
    x = descriptors.T.unsqueeze(0)  # (1, D, L)
    pooled = F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1.0 / p)
    return pooled.squeeze(0).squeeze(-1)  # (D,)


def _geo_distance_m(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * config.METERS_PER_DEGREE_LAT
    dlon = (lon2 - lon1) * config.METERS_PER_DEGREE_LON
    return (dlat ** 2 + dlon ** 2) ** 0.5


def _find_largest_cluster(frame_locations, radius_m):
    """Find the largest group of frame locations within radius_m of each other.
    frame_locations: list of (frame_idx, lat, lon)
    Returns: list of indices into frame_locations for the best cluster.
    """
    best_inliers = []
    for i, (_, lat_i, lon_i) in enumerate(frame_locations):
        inliers = []
        for j, (_, lat_j, lon_j) in enumerate(frame_locations):
            if _geo_distance_m(lat_i, lon_i, lat_j, lon_j) <= radius_m:
                inliers.append(j)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    return best_inliers


def query_video(video_path, sample_fps=config.VIDEO_SAMPLE_FPS,
                max_frames=config.VIDEO_MAX_FRAMES,
                top_k=config.TOP_K, do_rerank=False, crop_pct=None):
    device = get_device()

    frames = extract_frames(video_path, sample_fps, max_frames, crop_pct)
    if not frames:
        print("No frames extracted from video.")
        return

    ref_db = ReferenceDB()

    print(f"Loading SALAD model on {device}...")
    model = load_salad_model(device)
    transform = get_transform()

    tensors = torch.stack([transform(f) for f in frames])

    all_descriptors = []
    batch_size = min(config.BATCH_SIZE, 8)
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size].to(device)
        desc = model(batch)
        all_descriptors.append(desc.cpu())
    descriptors = torch.cat(all_descriptors, dim=0)  # (L, 8448)

    print(f"Computed {descriptors.shape[0]} frame descriptors ({descriptors.shape[1]}-dim)")

    frame_descriptors = descriptors.numpy().astype(np.float32)
    per_frame_k = max(top_k, config.PAIRVPR_CANDIDATES) if do_rerank else top_k
    distances, indices = ref_db.search(frame_descriptors, per_frame_k)

    frame_locations = []
    frame_results = []
    print(f"\nQuery: {video_path}")
    print(f"\nPer-frame top-1 matches:")
    for f_idx in range(len(frames)):
        results = ref_db.results(distances[f_idx], indices[f_idx])
        frame_results.append(results)

        if results:
            meta = results[0]["meta"]
            lat, lon = meta.get("lat"), meta.get("lon")
            if lat is not None and lon is not None:
                frame_locations.append((f_idx, lat, lon))
                print(f"  Frame {f_idx:2d}: lat={lat:.6f}, lon={lon:.6f}  d={distances[f_idx][0]:.4f}")
            else:
                print(f"  Frame {f_idx:2d}: match has no coordinates  d={distances[f_idx][0]:.4f}")
        else:
            print(f"  Frame {f_idx:2d}: no match")

    if len(frame_locations) < 2:
        print("Not enough frames with valid locations.")
        return

    best_inliers = _find_largest_cluster(frame_locations, CLUSTER_RADIUS_M)
    inlier_ratio = len(best_inliers) / len(frames)
    inlier_frame_indices = {frame_locations[i][0] for i in best_inliers}
    center_lat = np.mean([frame_locations[i][1] for i in best_inliers])
    center_lon = np.mean([frame_locations[i][2] for i in best_inliers])

    print(f"\nOOR check: {len(best_inliers)}/{len(frames)} frames cluster within {CLUSTER_RADIUS_M}m "
          f"({inlier_ratio:.0%}, threshold={MIN_INLIER_RATIO:.0%}, min count={MIN_INLIER_COUNT})")
    print(f"  Cluster center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"  Inlier frames: {[i for i in sorted(inlier_frame_indices)]}")

    if inlier_ratio < MIN_INLIER_RATIO or len(best_inliers) < MIN_INLIER_COUNT:
        print("OUT OF RANGE — no geographic consensus among frames.")
        return

    merged = {}
    for f_idx in sorted(inlier_frame_indices):
        for r in frame_results[f_idx][:top_k]:
            idx = r["idx"]
            if idx not in merged or r["distance"] < merged[idx]["distance"]:
                merged[idx] = r
    faiss_results = sorted(merged.values(), key=lambda r: r["distance"])

    print_results(f"Merged top {min(top_k, len(faiss_results))} from inlier frames:",
                  faiss_results[:top_k])

    faiss_lat, faiss_lon = weighted_centroid(faiss_results, CENTROID_TOP_M)
    if faiss_lat is not None:
        print(f"Weighted centroid (top-{CENTROID_TOP_M}): ({faiss_lat:.6f}, {faiss_lon:.6f})")

    if not do_rerank:
        return

    need_rerank = True
    if len(faiss_results) >= 2:
        d1, d2 = faiss_results[0]["distance"], faiss_results[1]["distance"]
        margin = (d2 - d1) / max(d1, 1e-6)
        print(f"\nFAISS margin (d2-d1)/d1 = {margin:.4f} (threshold={RERANK_MARGIN_THRESHOLD})")
        if margin > RERANK_MARGIN_THRESHOLD:
            print("High confidence — skipping Pair-VPR re-ranking.")
            need_rerank = False
        else:
            print("Low margin — triggering Pair-VPR re-ranking...")

    if not need_rerank:
        return

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\nLoading Pair-VPR for re-ranking...")
    from .pairvpr_utils import rerank_results

    best_frame_idx = sorted(inlier_frame_indices)[0]
    reranked_results = rerank_results(frames[best_frame_idx], faiss_results, device)

    print_results(f"\nPair-VPR re-ranked top {min(top_k, len(reranked_results))}:", reranked_results[:top_k])

    rerank_lat, rerank_lon = weighted_centroid(reranked_results, CENTROID_TOP_M)
    if rerank_lat is not None:
        print(f"Weighted centroid (top-{CENTROID_TOP_M}): ({rerank_lat:.6f}, {rerank_lon:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query VPR index with a video")
    parser.add_argument("video", help="Path to query video")
    parser.add_argument("--fps", type=float, default=config.VIDEO_SAMPLE_FPS,
                        help="Frame sampling rate (default: 2.0)")
    parser.add_argument("--max_frames", type=int, default=config.VIDEO_MAX_FRAMES,
                        help="Maximum frames to extract (default: 15)")
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument("--crop", type=str, default=None,
                        help="Crop top%%,bottom%% to remove HUD overlays (e.g. --crop 10,25)")
    parser.add_argument("--rerank", action="store_true",
                        help="Re-rank using Pair-VPR (adaptive)")
    args = parser.parse_args()

    crop_pct = None
    if args.crop:
        parts = args.crop.split(",")
        crop_pct = (float(parts[0]), float(parts[1]))

    query_video(args.video, args.fps, args.max_frames, args.top_k, args.rerank, crop_pct)
