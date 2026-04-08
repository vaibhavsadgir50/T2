"""
Build ``slot_init`` (n_slots × d_model) from Colab-saved knowledge graph vectors.

Pure data processing: no ``torch.nn``, no training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def _as_1d_int_degrees(x: Any, n: int) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        d = x.detach().cpu().numpy().reshape(-1)
    else:
        d = np.asarray(x, dtype=np.int64).reshape(-1)
    if d.shape[0] != n:
        raise ValueError(f"degrees length {d.shape[0]} != num vectors {n}")
    return d.astype(np.int64, copy=False)


def _degrees_from_edge_index(edge_index: torch.Tensor, n: int) -> np.ndarray:
    ei = edge_index.detach().cpu().long().numpy()
    if ei.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, E), got {tuple(edge_index.shape)}")
    deg = np.zeros(n, dtype=np.int64)
    src, dst = ei[0], ei[1]
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    return deg


def _degrees_from_adjacency(adj: torch.Tensor, n: int) -> np.ndarray:
    a = adj.detach().cpu().float().numpy()
    if a.shape != (n, n):
        raise ValueError(f"adjacency must be ({n}, {n}), got {tuple(a.shape)}")
    return (a != 0).astype(np.float64).sum(axis=1).astype(np.int64)


def _degrees_from_edges(edges: Sequence, n: int) -> np.ndarray:
    deg = np.zeros(n, dtype=np.int64)
    for e in edges:
        if len(e) != 2:
            raise ValueError(f"each edge must be (i, j), got {e!r}")
        i, j = int(e[0]), int(e[1])
        deg[i] += 1
        deg[j] += 1
    return deg


def _mutual_knn_degrees(vectors_np: np.ndarray, k: int = 32) -> np.ndarray:
    """Undirected mutual k-NN graph on L2-normalized rows; degree = neighbor count."""
    n = vectors_np.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    x = np.asarray(vectors_np, dtype=np.float64)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms
    xt = torch.from_numpy(x)
    sim = (xt @ xt.T).numpy()
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(k, max(1, n - 1))
    knn = np.argsort(-sim, axis=1)[:, :k_eff]
    is_neighbor = np.zeros((n, n), dtype=bool)
    rows = np.arange(n, dtype=np.int64)[:, None]
    is_neighbor[rows, knn] = True
    mutual = is_neighbor & is_neighbor.T
    np.fill_diagonal(mutual, False)
    return mutual.sum(axis=1).astype(np.int64)


def _resolve_degrees(data: Dict[str, Any], vectors: torch.Tensor) -> np.ndarray:
    n = vectors.shape[0]
    if "degrees" in data:
        return _as_1d_int_degrees(data["degrees"], n)
    if "degree" in data:
        return _as_1d_int_degrees(data["degree"], n)
    if "node_degrees" in data:
        return _as_1d_int_degrees(data["node_degrees"], n)
    if "edge_index" in data:
        return _degrees_from_edge_index(data["edge_index"], n)
    if "adjacency" in data:
        return _degrees_from_adjacency(data["adjacency"], n)
    if "edges" in data:
        return _degrees_from_edges(data["edges"], n)
    return _mutual_knn_degrees(vectors.detach().cpu().numpy())


def _l2_normalize_rows(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = t.norm(dim=-1, keepdim=True).clamp_min(eps)
    return t / n


def _kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """x: (n, d) float32, returns (k, d) centroids."""
    n, d = x.shape
    centroids = np.empty((k, d), dtype=np.float32)
    idx0 = int(rng.integers(0, n))
    centroids[0] = x[idx0]
    closest = np.sum((x - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest / (closest.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centroids[i] = x[idx]
        dist = np.sum((x - centroids[i]) ** 2, axis=1)
        closest = np.minimum(closest, dist)
    return centroids


def _kmeans_lloyd(
    x: np.ndarray, k: int, max_iter: int = 100, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lloyd on rows of x. Returns (centroids (k, d), labels (n,) int).
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    if k > n:
        raise ValueError(f"kmeans k={k} cannot exceed n={n}")
    centroids = _kmeans_plus_plus_init(x, k, rng)
    labels = np.zeros(n, dtype=np.int64)
    x64 = torch.from_numpy(x.astype(np.float64, copy=False))
    c64 = torch.from_numpy(centroids.astype(np.float64, copy=False))
    for _ in range(max_iter):
        dists = (
            torch.sum(x64[:, None, :] ** 2, dim=2)
            + torch.sum(c64[None, :, :] ** 2, dim=2)
            - 2 * (x64 @ c64.T)
        ).numpy()
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centroids[j] = x[mask].mean(axis=0).astype(np.float32, copy=False)
            else:
                idx = int(rng.integers(0, n))
                centroids[j] = x[idx]
        c64 = torch.from_numpy(centroids.astype(np.float64, copy=False))
    return centroids, labels


def _nearest_original_to_centroids(
    x: np.ndarray, centroids: np.ndarray, labels: np.ndarray, k: int
) -> np.ndarray:
    """One representative index per cluster: row closest to centroid (by L2)."""
    picked = np.empty(k, dtype=np.int64)
    for j in range(k):
        mask = labels == j
        if not np.any(mask):
            picked[j] = int(np.argmin(np.sum((x - centroids[j]) ** 2, axis=1)))
            continue
        idxs = np.where(mask)[0]
        sub = x[idxs]
        c = centroids[j]
        d2 = np.sum((sub - c) ** 2, axis=1)
        picked[j] = int(idxs[int(np.argmin(d2))])
    return picked


def _concept_names_for_indices(
    data: Dict[str, Any], indices: np.ndarray
) -> List[str]:
    names: Optional[Sequence[str]] = None
    if "top_words" in data:
        names = data["top_words"]
    elif "concept_names" in data:
        names = data["concept_names"]
    if names is None:
        return [str(int(i)) for i in indices.tolist()]
    names_list = list(names)
    if len(names_list) < max(indices) + 1:
        raise ValueError(
            f"concept name list length {len(names_list)} < max index {int(indices.max()) + 1}"
        )
    return [str(names_list[int(i)]) for i in indices.tolist()]


def build_slot_init(
    graph_vectors_path: Union[str, Path],
    output_path: Union[str, Path],
    n_slots: int = 64,
    strategy: str = "degree",
    d_model: int = 300,
) -> Dict[str, Any]:
    """
    Load graph vectors, pick ``n_slots`` concepts, L2-normalize, save ``slot_init.pt``.

    ``graph_vectors.pt`` must contain at least ``vectors`` (N, D) and usually ``top_words``
    (length N). Optional graph fields for ``strategy="degree"``: ``degrees``, ``edge_index``,
    ``adjacency``, or ``edges``. If none are present, degree is derived from a mutual k-NN
    graph on normalized vectors.
    """
    if strategy not in ("degree", "diverse"):
        raise ValueError(f"strategy must be 'degree' or 'diverse', got {strategy!r}")

    path = Path(graph_vectors_path)
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"expected dict in {path}, got {type(data)}")

    if "vectors" not in data:
        raise KeyError(f"{path} must contain key 'vectors'")

    vectors = data["vectors"]
    if not isinstance(vectors, torch.Tensor):
        vectors = torch.as_tensor(vectors, dtype=torch.float32)
    else:
        vectors = vectors.float().cpu()

    if vectors.dim() != 2:
        raise ValueError(f"vectors must be 2D, got shape {tuple(vectors.shape)}")
    n, d = vectors.shape
    if d != d_model:
        raise ValueError(f"vectors dim {d} != d_model {d_model}")
    if n < n_slots:
        raise ValueError(f"need at least n_slots={n_slots} vectors, got n={n}")

    file_dim = data.get("dim")
    if file_dim is not None and int(file_dim) != d_model:
        raise ValueError(f"file dim={file_dim} != d_model={d_model}")

    if strategy == "degree":
        deg = _resolve_degrees(data, vectors)
        order = np.argsort(-deg, kind="stable")
        indices = order[:n_slots]
    else:
        x = vectors.detach().cpu().numpy().astype(np.float32, copy=False)
        xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
        centroids, labels = _kmeans_lloyd(xn, n_slots)
        indices = _nearest_original_to_centroids(xn, centroids, labels, n_slots)

    idx_t = torch.as_tensor(indices, dtype=torch.long)
    chosen = vectors[idx_t].clone()
    chosen = _l2_normalize_rows(chosen)
    concept_names = _concept_names_for_indices(data, indices)

    bundle: Dict[str, Any] = {
        "slot_init": chosen,
        "concept_names": concept_names,
        "strategy": strategy,
        "n_slots": n_slots,
        "d_model": d_model,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)

    print(f"Saved {out}")
    print(f"Strategy: {strategy}  |  n_slots={n_slots}  d_model={d_model}")
    print("Selected concepts:")
    for i, name in enumerate(concept_names):
        print(f"  [{i:2d}] {name}")

    return bundle


def load_slot_init(path: Union[str, Path], device: str = "cpu") -> torch.Tensor:
    """Load ``slot_init.pt`` and return the ``(n_slots, d_model)`` tensor only."""
    p = Path(path)
    try:
        data = torch.load(p, map_location=device, weights_only=False)
    except TypeError:
        data = torch.load(p, map_location=device)
    if not isinstance(data, dict) or "slot_init" not in data:
        raise KeyError(f"{p} must be a dict with key 'slot_init'")
    t = data["slot_init"]
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.float32, device=device)
    else:
        t = t.to(device=device)
    if t.dim() != 2:
        raise ValueError(f"slot_init must be 2D, got {tuple(t.shape)}")
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description="Build slot_init.pt from graph vectors.")
    parser.add_argument(
        "--graph_vectors",
        type=str,
        required=True,
        help="Path to graph_vectors.pt (Colab Cell 4 output).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write slot_init.pt.",
    )
    parser.add_argument("--n_slots", type=int, default=64)
    parser.add_argument(
        "--strategy",
        type=str,
        default="degree",
        choices=("degree", "diverse"),
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=300,
        help="Expected embedding dim (must match vectors and v2/config N_EMBD).",
    )
    args = parser.parse_args()
    build_slot_init(
        args.graph_vectors,
        args.output,
        n_slots=args.n_slots,
        strategy=args.strategy,
        d_model=args.d_model,
    )


if __name__ == "__main__":
    main()
