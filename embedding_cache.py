# EMBEDDING CACHE - persistent sentence-transformer embedding cache
#
# Embedding tens of thousands of articles is the slow part of the pipeline
# (~15-20 min for the 32k emerging corpus on CPU). Article text rarely changes
# between runs, so we cache each text's embedding to disk keyed by a hash of
# (model name + text). Re-runs then embed only the *new* articles.
#
# The cache is a single compressed .npz-per-shard store is overkill here;
# instead we keep one memory-mapped .npy matrix plus a JSON key->row index,
# which is compact and fast to append to.
#
# Usage:
#   from embedding_cache import encode_cached
#   vecs = encode_cached(model, texts, model_name="all-MiniLM-L6-v2")
#
# `vecs` is a float32 np.ndarray of shape (len(texts), dim), L2-normalized,
# in the same order as `texts`.

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).parent / "output" / ".emb_cache"


def _key(model_name, text):
    h = hashlib.sha1()
    h.update((model_name or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _store_paths(model_name):
    """Per-model store: a vectors matrix + an index mapping key -> row."""
    safe = "".join(c if c.isalnum() else "_" for c in (model_name or "model"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{safe}.vecs.npy", CACHE_DIR / f"{safe}.index.json"


def _load_store(model_name):
    vecs_path, index_path = _store_paths(model_name)
    if vecs_path.exists() and index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            vecs = np.load(vecs_path)
            if len(index) == len(vecs):
                return index, vecs, vecs_path, index_path
        except Exception:
            pass  # corrupt/mismatched cache -> rebuild
    return {}, None, vecs_path, index_path


def encode_cached(model, texts, model_name=None, batch_size=64, normalize=True,
                  verbose=True):
    """Encode `texts`, reusing cached vectors and embedding only cache misses.

    Parameters
    ----------
    model : SentenceTransformer  (only .encode is used; only called on misses)
    texts : list[str]
    model_name : str  cache namespace; should identify the model/weights.

    Returns float32 ndarray (len(texts), dim), L2-normalized if `normalize`.
    """
    texts = [("" if t is None else str(t)) for t in texts]
    if model_name is None:
        model_name = getattr(model, "_model_card_vars", {}).get("name") or \
            model.__class__.__name__

    index, vecs, vecs_path, index_path = _load_store(model_name)

    # Determine misses (unique texts not already cached).
    keys = [_key(model_name, t) for t in texts]
    unique_missing = {}
    for k, t in zip(keys, texts):
        if k not in index and k not in unique_missing:
            unique_missing[k] = t

    hits = len(set(keys)) - len(unique_missing)
    if verbose:
        print(f"  embedding cache: {hits} hit(s), {len(unique_missing)} miss(es) "
              f"[{len(set(keys))} unique of {len(texts)} texts]")

    # Embed misses in one batch and append to the store.
    if unique_missing:
        miss_keys = list(unique_missing.keys())
        miss_texts = [unique_missing[k] for k in miss_keys]
        new_vecs = model.encode(
            miss_texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=normalize, show_progress_bar=verbose,
        ).astype(np.float32)

        if vecs is None:
            vecs = new_vecs
            start = 0
        else:
            start = len(vecs)
            vecs = np.vstack([vecs, new_vecs])
        for i, k in enumerate(miss_keys):
            index[k] = start + i

        # Persist (best-effort; a failed write just means we re-embed next time).
        try:
            np.save(vecs_path, vecs)
            index_path.write_text(json.dumps(index), encoding="utf-8")
        except Exception as e:
            if verbose:
                print(f"  WARNING: could not persist embedding cache: {e}")

    # Assemble the output in the original text order.
    dim = vecs.shape[1]
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, k in enumerate(keys):
        out[i] = vecs[index[k]]
    return out


def clear_cache(model_name=None):
    """Delete cached vectors (all models, or one). Returns files removed."""
    removed = 0
    if not CACHE_DIR.exists():
        return 0
    if model_name:
        for p in _store_paths(model_name):
            if p.exists():
                p.unlink()
                removed += 1
    else:
        for p in CACHE_DIR.glob("*"):
            if p.is_file():
                p.unlink()
                removed += 1
    return removed


if __name__ == "__main__":
    # smoke test with a tiny fake model
    class _Fake:
        def encode(self, texts, **kw):
            # deterministic pseudo-embedding from text length + first char
            arr = np.array([[len(t), ord(t[0]) if t else 0, 1.0] for t in texts],
                           dtype=np.float32)
            n = np.linalg.norm(arr, axis=1, keepdims=True)
            return arr / np.where(n == 0, 1, n)

    m = _Fake()
    t1 = ["alpha", "beta", "gamma"]
    v1 = encode_cached(m, t1, model_name="_smoke")
    v2 = encode_cached(m, t1 + ["delta"], model_name="_smoke")  # 3 hits, 1 miss
    print("shapes:", v1.shape, v2.shape)
    print("stable rows match:", np.allclose(v1, v2[:3]))
    print("removed:", clear_cache("_smoke"))
