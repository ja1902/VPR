"""
Build the search index from extracted SALAD descriptors.

The FAISS index stores compressed vectors and only produces a shortlist, which
vpr.search.ReferenceDB re-scores exactly against the full descriptors on disk.
Index type is chosen from the database size (override with INDEX_FACTORY):

    fewer than 4 x PCA dim vectors  -> Flat               (tiny database, no compression)
    fewer than INDEX_IVF_MIN_VECTORS -> PCA2048,SQ8        (exhaustive scan, 2 KB/vector)
    larger                          -> PCA2048,IVFn,SQ8   (inverted lists, scans nprobe/n)

Training uses a random sample and vectors are added in chunks, so RAM stays bounded
for city-scale databases. Per-image metadata is written to SQLite.

Usage:
    python -m vpr.build_index
    python -m vpr.build_index --eval 200   # also report shortlist recall vs exact search
"""
import argparse
import json
import math
import pickle
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from . import config
from .search import ReferenceDB


def choose_factory(n):
    if config.INDEX_FACTORY:
        return config.INDEX_FACTORY
    if n < 4 * config.INDEX_PCA_DIM:
        return "Flat"
    pca = f"PCA{config.INDEX_PCA_DIM}"
    if n < config.INDEX_IVF_MIN_VECTORS:
        return f"{pca},SQ8"
    n_train = min(n, config.INDEX_TRAIN_SAMPLES)
    nlist = min(int(4 * math.sqrt(n)), n_train // 39)
    return f"{pca},IVF{nlist},SQ8"


def write_reference_db(image_paths, pano_metadata):
    # Each perspective image path looks like: .../perspectives/{pano_id}_{direction}.jpg
    def rows():
        for i, img_path in enumerate(image_paths):
            pano_id, direction = Path(img_path).stem.rsplit("_", 1)
            meta = pano_metadata.get(pano_id, {})
            yield (i, img_path, pano_id, direction, meta.get("lat"), meta.get("lon"),
                   meta.get("heading"), meta.get("date"))

    config.REFERENCE_DB_FILE.unlink(missing_ok=True)
    db = sqlite3.connect(str(config.REFERENCE_DB_FILE))
    with db:
        db.execute(
            "CREATE TABLE refs (id INTEGER PRIMARY KEY, image_path TEXT, pano_id TEXT, "
            "direction TEXT, lat REAL, lon REAL, heading REAL, date TEXT)"
        )
        db.executemany("INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows())
    db.close()


def train_pca(pca, x):
    """Fit a faiss PCAMatrix with numpy. faiss' own PCA training runs a single-threaded
    eigendecomposition that takes >10 min at d=8448; multithreaded eigh takes ~1 min."""
    mean = x.mean(axis=0)
    cov = (x.T @ x) / len(x)
    cov -= np.outer(mean, mean)  # centre without allocating a centred copy of x
    eigvals, eigvecs = np.linalg.eigh(cov)
    top = np.argsort(eigvals)[::-1][:pca.d_out]
    faiss.copy_array_to_vector(mean.astype(np.float32), pca.mean)
    faiss.copy_array_to_vector(eigvals[top].astype(np.float32), pca.eigenvalues)
    faiss.copy_array_to_vector(np.ascontiguousarray(eigvecs[:, top].T, dtype=np.float32).ravel(), pca.PCAMat)
    pca.prepare_Ab()
    pca.PCAMat.resize(0)  # only A/b are used to apply the transform; don't store the matrix twice
    pca.is_trained = True


def train_index(index, sample):
    if isinstance(index, faiss.IndexPreTransform):
        for i in range(index.chain.size()):
            vt = faiss.downcast_VectorTransform(index.chain.at(i))
            if type(vt) is faiss.PCAMatrix and vt.eigen_power == 0 and not vt.random_rotation:
                train_pca(vt, sample)
            elif not vt.is_trained:
                vt.train(sample)
            sample = vt.apply(sample)
        sub_index = faiss.downcast_index(index.index)
        if not sub_index.is_trained:
            sub_index.train(sample)
        index.is_trained = True
    else:
        index.train(sample)


def build_faiss_index(descriptors, factory):
    n, d = descriptors.shape
    index = faiss.index_factory(d, factory, faiss.METRIC_L2)

    if not index.is_trained:
        n_train = min(n, config.INDEX_TRAIN_SAMPLES)
        rng = np.random.default_rng(0)
        sample = np.sort(rng.choice(n, n_train, replace=False))
        print(f"Training on {n_train} sampled vectors...")
        train_index(index, np.ascontiguousarray(descriptors[sample], dtype=np.float32))

    for start in range(0, n, config.INDEX_ADD_CHUNK):
        chunk = descriptors[start:start + config.INDEX_ADD_CHUNK]
        index.add(np.ascontiguousarray(chunk, dtype=np.float32))
    return index


def shortlist_recall(ref_db, descriptors, n_queries, k=10):
    """Recall@k of ReferenceDB.search vs exact brute force, using reference vectors as
    queries (excluding the self-match)."""
    n = len(descriptors)
    rng = np.random.default_rng(1)
    qids = rng.choice(n, min(n_queries, n), replace=False)
    queries = np.asarray(descriptors[qids], dtype=np.float32)

    # Exact squared L2, streamed over the memmap in chunks
    exact_d = np.full((len(qids), k + 1), np.inf, dtype=np.float32)
    exact_i = np.full((len(qids), k + 1), -1, dtype=np.int64)
    q_norm = (queries ** 2).sum(1, keepdims=True)
    for start in range(0, n, config.INDEX_ADD_CHUNK):
        chunk = np.asarray(descriptors[start:start + config.INDEX_ADD_CHUNK], dtype=np.float32)
        d = q_norm + (chunk ** 2).sum(1)[None, :] - 2 * queries @ chunk.T
        all_d = np.concatenate([exact_d, d], axis=1)
        all_i = np.concatenate([exact_i, np.broadcast_to(np.arange(start, start + len(chunk)), d.shape)], axis=1)
        top = np.argsort(all_d, axis=1)[:, :k + 1]
        exact_d = np.take_along_axis(all_d, top, 1)
        exact_i = np.take_along_axis(all_i, top, 1)

    _, approx_i = ref_db.search(queries, k + 1)
    hits = 0
    for qid, ex, ap in zip(qids, exact_i, approx_i):
        ex = [i for i in ex if i != qid][:k]
        ap = [i for i in ap if i != qid][:k]
        hits += len(set(ex) & set(ap))
    return hits / (len(qids) * k)


def main():
    parser = argparse.ArgumentParser(description="Build the VPR search index")
    parser.add_argument("--eval", type=int, default=0, metavar="N",
                        help="Measure shortlist recall@10 vs exact search on N sample queries")
    args = parser.parse_args()

    list_path = config.FEATURES_DIR / "image_list.pkl"

    if not config.DESCRIPTORS_FILE.exists():
        print(f"Descriptors not found at {config.DESCRIPTORS_FILE}")
        print("Run python -m vpr.extract_features first.")
        return

    descriptors = np.load(config.DESCRIPTORS_FILE, mmap_mode="r")
    with open(list_path, "rb") as f:
        image_paths = pickle.load(f)

    with open(config.METADATA_FILE) as f:
        pano_metadata = json.load(f)

    n, d = descriptors.shape
    factory = choose_factory(n)
    print(f"Building index: {n} vectors, {d} dimensions, factory=\"{factory}\"")

    index = build_faiss_index(descriptors, factory)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_FILE))
    write_reference_db(image_paths, pano_metadata)

    index_mb = config.INDEX_FILE.stat().st_size / 2**20
    flat_mb = n * d * 4 / 2**20
    print(f"Index saved to {config.INDEX_FILE} ({index_mb:.1f} MB in RAM at query time; "
          f"an uncompressed FlatL2 index would be {flat_mb:.1f} MB)")
    print(f"Reference metadata saved to {config.REFERENCE_DB_FILE} ({n} entries)")

    # Sanity check: query the first vector against itself through the full search path
    ref_db = ReferenceDB()
    D, I = ref_db.search(np.asarray(descriptors[:1], dtype=np.float32), 1)
    print(f"\nSanity check: nearest neighbor of vector 0 is vector {I[0][0]} (distance: {D[0][0]:.4f})")
    if I[0][0] == 0 and D[0][0] < 1e-6:
        print("  PASS: self-match confirmed")
    else:
        print("  WARNING: self-match failed — check descriptor extraction")

    if args.eval:
        recall = shortlist_recall(ref_db, descriptors, args.eval)
        print(f"\nRecall@10 vs exact search ({args.eval} queries): {recall:.3f}")


if __name__ == "__main__":
    main()
