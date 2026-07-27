"""
Build a FAISS index from extracted SALAD descriptors.
Uses IndexFlatL2 wrapped in IndexIDMap for prototype scale (<10k vectors).
"""
import json
import pickle

import faiss
import numpy as np

from . import config


def main():
    desc_path = config.FEATURES_DIR / "descriptors.npy"
    list_path = config.FEATURES_DIR / "image_list.pkl"
    metadata_path = config.METADATA_FILE

    if not desc_path.exists():
        print(f"Descriptors not found at {desc_path}")
        print("Run 04_extract_features.py first.")
        return

    descriptors = np.load(desc_path).astype(np.float32)
    with open(list_path, "rb") as f:
        image_paths = pickle.load(f)

    with open(metadata_path) as f:
        pano_metadata = json.load(f)

    print(f"Building index: {descriptors.shape[0]} vectors, {descriptors.shape[1]} dimensions")

    # Build ID-to-metadata mapping
    # Each perspective image path looks like: .../perspectives/{pano_id}_{direction}.jpg
    id_map = {}
    for i, img_path in enumerate(image_paths):
        filename = img_path.split("/")[-1]  # e.g. "abc123_N.jpg"
        stem = filename.rsplit(".", 1)[0]    # e.g. "abc123_N"
        pano_id = stem.rsplit("_", 1)[0]     # e.g. "abc123"
        direction = stem.rsplit("_", 1)[1]    # e.g. "N"

        meta = pano_metadata.get(pano_id, {})
        id_map[i] = {
            "image_path": img_path,
            "pano_id": pano_id,
            "direction": direction,
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            "heading": meta.get("heading"),
            "date": meta.get("date"),
        }

    # Build FAISS index
    quantizer = faiss.IndexFlatL2(descriptors.shape[1])
    index = faiss.IndexIDMap(quantizer)

    ids = np.arange(len(descriptors), dtype=np.int64)
    index.add_with_ids(descriptors, ids)

    # Save
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = str(config.INDEX_DIR / "region.index")
    faiss.write_index(index, index_path)

    id_map_path = config.INDEX_DIR / "id_map.pkl"
    with open(id_map_path, "wb") as f:
        pickle.dump(id_map, f)

    print(f"Index saved to {index_path}")
    print(f"ID map saved to {id_map_path} ({len(id_map)} entries)")

    # Sanity check: query the first vector against itself
    D, I = index.search(descriptors[:1], 1)
    print(f"\nSanity check: nearest neighbor of vector 0 is vector {I[0][0]} (distance: {D[0][0]:.4f})")
    if I[0][0] == 0 and D[0][0] < 1e-6:
        print("  PASS: self-match confirmed")
    else:
        print("  WARNING: self-match failed — check descriptor extraction")


if __name__ == "__main__":
    main()
