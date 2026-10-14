"""
Reference database used at query time.

- FAISS index of compressed descriptors (PCA + 8-bit scalar quantization by default),
  used only to produce a shortlist of candidates. It is ~30x smaller than a raw
  8448-dim float32 FlatL2 index.
- Full descriptors (float16 .npy), memory-mapped from disk. Only the shortlisted rows
  are read, and they are re-scored with exact squared L2, so returned distances have the
  same scale as a FlatL2 search (the re-rank margin threshold still applies).
- Per-image metadata in SQLite, looked up by id instead of unpickling a dict of every
  reference image into RAM.
"""
import sqlite3

import faiss
import numpy as np

from . import config


class ReferenceDB:
    def __init__(self):
        missing = [p for p in (config.INDEX_FILE, config.REFERENCE_DB_FILE, config.DESCRIPTORS_FILE)
                   if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {', '.join(str(p) for p in missing)}. Run: python -m vpr.build_index"
            )

        self.index = faiss.read_index(str(config.INDEX_FILE))
        ivf = faiss.try_extract_index_ivf(self.index)
        if ivf is not None:
            ivf.nprobe = config.INDEX_NPROBE

        self.descriptors = np.load(config.DESCRIPTORS_FILE, mmap_mode="r")
        self._db = sqlite3.connect(str(config.REFERENCE_DB_FILE))
        self._db.row_factory = sqlite3.Row
        self._meta_cache = {}

    def __len__(self):
        return self.index.ntotal

    def search(self, queries, k):
        """Exact top-k (squared L2, ascending) among a shortlist from the compressed index.

        queries: [nq, D] float array. Returns (distances [nq, k], ids [nq, k]); missing
        entries are padded with inf / -1, like FAISS.
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        shortlist = min(max(k, config.SHORTLIST_K), self.index.ntotal)
        _, candidates = self.index.search(queries, shortlist)

        distances = np.full((len(queries), k), np.inf, dtype=np.float32)
        ids = np.full((len(queries), k), -1, dtype=np.int64)
        for qi, (q, cand) in enumerate(zip(queries, candidates)):
            cand = np.unique(cand[cand >= 0])  # sorted ids -> mostly sequential disk reads
            refs = self.descriptors[cand].astype(np.float32)
            d = ((refs - q) ** 2).sum(axis=1)
            order = np.argsort(d)[:k]
            distances[qi, :len(order)] = d[order]
            ids[qi, :len(order)] = cand[order]
        return distances, ids

    def meta(self, idx):
        if idx not in self._meta_cache:
            row = self._db.execute(
                "SELECT image_path, pano_id, direction, lat, lon, heading, date FROM refs WHERE id = ?",
                (idx,),
            ).fetchone()
            self._meta_cache[idx] = dict(row)
        return self._meta_cache[idx]

    def results(self, distances, ids):
        """Turn one row of search output into the result dicts used by the query scripts."""
        return [
            {"distance": float(d), "meta": self.meta(int(i)), "idx": int(i)}
            for d, i in zip(distances, ids) if i != -1
        ]
