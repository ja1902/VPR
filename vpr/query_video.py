"""
Query the FAISS index with a video to find the closest matching location.
Extracts frames at a configurable FPS, computes per-frame SALAD descriptors,
applies SeqGeM temporal pooling to produce a single 8448-dim query vector,
then runs FAISS search with optional adaptive Pair-VPR re-ranking.

Usage:
    python 07_query_video.py <video_path>
    python 07_query_video.py <video_path> --rerank
    python 07_query_video.py <video_path> --fps 3 --max_frames 10 --rerank
    python 07_query_video.py <video_path> --crop 10,25 --rerank   # dashcam: trim 10% top, 25% bottom
"""
import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import faiss
import cv2
from PIL import Image

from . import config
from .model_utils import load_salad_model, get_transform
from .query_image import weighted_centroid, print_results, RERANK_MARGIN_THRESHOLD, CENTROID_TOP_M


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


def query_video(video_path, sample_fps=config.VIDEO_SAMPLE_FPS,
                max_frames=config.VIDEO_MAX_FRAMES,
                top_k=config.TOP_K, do_rerank=False, crop_pct=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = extract_frames(video_path, sample_fps, max_frames, crop_pct)
    if not frames:
        print("No frames extracted from video.")
        return

    index_path = str(config.INDEX_DIR / "region.index")
    index = faiss.read_index(index_path)

    with open(config.INDEX_DIR / "id_map.pkl", "rb") as f:
        id_map = pickle.load(f)

    print(f"Loading SALAD model on {device}...")
    model = load_salad_model(device)
    transform = get_transform()

    tensors = torch.stack([transform(f) for f in frames])

    with torch.no_grad():
        all_descriptors = []
        batch_size = min(config.BATCH_SIZE, 8)
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size].to(device)
            desc = model(batch)
            all_descriptors.append(desc.cpu())
        descriptors = torch.cat(all_descriptors, dim=0)  # (L, 8448)

    print(f"Computed {descriptors.shape[0]} frame descriptors ({descriptors.shape[1]}-dim)")

    if descriptors.shape[0] == 1:
        print("Single frame — skipping temporal pooling")
        query_descriptor = descriptors.numpy().astype(np.float32)
    else:
        pooled = seqgem_pool(descriptors, p=3.0)
        query_descriptor = pooled.unsqueeze(0).numpy().astype(np.float32)
        print(f"SeqGeM pooled {descriptors.shape[0]} frames → single {query_descriptor.shape[1]}-dim descriptor")

    retrieval_k = max(top_k, 100) if do_rerank else top_k
    distances, indices = index.search(query_descriptor, retrieval_k)

    faiss_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = id_map[int(idx)]
        faiss_results.append({"distance": float(dist), "meta": meta, "idx": int(idx)})

    print(f"\nQuery: {video_path}")
    print_results(f"FAISS top {min(top_k, len(faiss_results))} matches:", faiss_results[:top_k], id_map)

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
    from .pairvpr_utils import load_pair_vpr_model, extract_dense_features, rerank

    pair_model = load_pair_vpr_model(device)
    pair_transform = get_transform()

    best_frame = frames[0]
    query_tensor = pair_transform(best_frame).unsqueeze(0)
    query_feats, _ = extract_dense_features(pair_model, query_tensor, device)

    candidate_feats = []
    for result in faiss_results:
        cand_img = Image.open(result["meta"]["image_path"]).convert("RGB")
        cand_tensor = pair_transform(cand_img).unsqueeze(0)
        cand_feat, _ = extract_dense_features(pair_model, cand_tensor, device)
        candidate_feats.append(cand_feat.squeeze(0))

    candidate_feats = torch.stack(candidate_feats)
    sorted_indices, sorted_scores = rerank(pair_model, query_feats, candidate_feats, device)

    reranked_results = []
    for rerank_idx, score in zip(sorted_indices, sorted_scores):
        original = faiss_results[rerank_idx.item()]
        reranked_results.append({
            "score": score.item(),
            "meta": original["meta"],
            "original_rank": rerank_idx.item() + 1,
        })

    print_results(f"\nPair-VPR re-ranked top {min(top_k, len(reranked_results))}:", reranked_results[:top_k], id_map)

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
