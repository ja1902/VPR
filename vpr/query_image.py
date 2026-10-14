"""
Query the FAISS index with a photo to find the closest matching location.
Adaptive re-ranking: Pair-VPR is triggered only when FAISS top-1 and top-2
scores are close (margin below threshold). Final coordinates are resolved
via weighted centroid interpolation over the top-M matches.

Usage:
    python 06_query.py <image_path>
    python 06_query.py <image_path> --rerank
    python 06_query.py <image_path> --top_k 20 --rerank
"""
import argparse

import numpy as np
import torch
from PIL import Image

from . import config
from .model_utils import load_salad_model, get_transform, get_device
from .search import ReferenceDB

RERANK_MARGIN_THRESHOLD = 0.15
OOR_PAIRVPR_THRESHOLD = 0.0
CENTROID_TOP_M = 5


def weighted_centroid(ranked_items, top_m=CENTROID_TOP_M):
    items = ranked_items[:top_m]
    lats, lons, weights = [], [], []
    for item in items:
        lat = item["meta"].get("lat")
        lon = item["meta"].get("lon")
        if lat is None or lon is None:
            continue
        if "score" in item:
            w = max(item["score"], 0.01)
        else:
            w = 1.0 / max(item["distance"], 1e-6)
        lats.append(lat)
        lons.append(lon)
        weights.append(w)

    if not weights:
        return None, None

    total_w = sum(weights)
    avg_lat = sum(la * w for la, w in zip(lats, weights)) / total_w
    avg_lon = sum(lo * w for lo, w in zip(lons, weights)) / total_w
    return avg_lat, avg_lon


def print_results(header, ranked_items):
    print(f"\n{header}")
    print("-" * 80)
    for rank, item in enumerate(ranked_items, 1):
        meta = item["meta"]
        lat = meta.get("lat", "?")
        lon = meta.get("lon", "?")
        score_str = f"distance={item['distance']:.4f}" if "distance" in item else f"score={item['score']:.4f}"
        print(
            f"  {rank}. {score_str}  "
            f"lat={lat}, lon={lon}  "
            f"pano={meta['pano_id']}_{meta['direction']}  "
            f"date={meta.get('date', '?')}"
        )
    print("-" * 80)
    best = ranked_items[0]["meta"]
    print(f"\nTop-1 match: ({best.get('lat')}, {best.get('lon')})")
    print(f"Reference image: {best['image_path']}")


def query(image_path, top_k=config.TOP_K, do_rerank=False):
    device = get_device()
    ref_db = ReferenceDB()

    print(f"Loading SALAD model on {device}...")
    model = load_salad_model(device)
    transform = get_transform()

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    query_descriptor = model(tensor).cpu().numpy().astype(np.float32)

    retrieval_k = max(top_k, config.PAIRVPR_CANDIDATES) if do_rerank else top_k
    distances, indices = ref_db.search(query_descriptor, retrieval_k)
    faiss_results = ref_db.results(distances[0], indices[0])

    print(f"\nQuery: {image_path}")
    print_results(f"FAISS top {min(top_k, len(faiss_results))} matches:", faiss_results[:top_k])

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

    reranked_results = rerank_results(img, faiss_results, device)

    print_results(f"\nPair-VPR re-ranked top {min(top_k, len(reranked_results))}:", reranked_results[:top_k])

    best_score = reranked_results[0]["score"]
    print(f"\nOOR check: best Pair-VPR score={best_score:.4f} (threshold={OOR_PAIRVPR_THRESHOLD})")
    if best_score < OOR_PAIRVPR_THRESHOLD:
        print("OUT OF RANGE — Pair-VPR confirms no good match exists.")
        return

    rerank_lat, rerank_lon = weighted_centroid(reranked_results, CENTROID_TOP_M)
    if rerank_lat is not None:
        print(f"Weighted centroid (top-{CENTROID_TOP_M}): ({rerank_lat:.6f}, {rerank_lon:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query VPR index with an image")
    parser.add_argument("image", help="Path to query image")
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument("--rerank", action="store_true", help="Re-rank using Pair-VPR (adaptive: only when margin is low)")
    args = parser.parse_args()

    query(args.image, args.top_k, args.rerank)
