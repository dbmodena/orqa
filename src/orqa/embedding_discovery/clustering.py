"""
Cluster-based neighbor nomination over the metadata embeddings.

Datasets are grouped by cosine-KMeans clustering; a dataset near a cluster
boundary also joins any OTHER cluster whose centroid it's almost as close
to, so cross-cluster joins/unions/correlations aren't cut off by a hard
partition. Oversized clusters are capped so no single cluster explodes the
number of pairs a dataset must be checked against.

Exposes the same ``query_neighbors(dataset_id, top_k, cos_threshold)``
interface the discovery pipeline used against the earlier HNSW index, so
clustering is a drop-in swap for the neighbor source.
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..utils.pipeline_logger import PipelineLogger

logger = logging.getLogger(__name__)


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_cluster_neighbors(
    ids: list[str],
    vectors: np.ndarray,
    target_cluster_size: int,
    overlap_margin: float,
    max_cluster_size: int,
    seed: int,
) -> dict[str, list[str]]:
    """Cluster datasets and return each dataset's cluster-mates.

    ``n_clusters`` is derived from ``target_cluster_size`` so it scales with
    the portal instead of needing per-city tuning. Membership is soft: a
    point joins every cluster whose centroid similarity is within
    ``overlap_margin`` (cosine units) of its own cluster's similarity, not
    only its single nearest centroid.
    """
    n = len(ids)
    log = PipelineLogger()
    if n < 2:
        return {i: [] for i in ids}

    n_clusters = max(1, round(n / max(1, target_cluster_size)))
    log.clustering_start(n, target_cluster_size)
    unit = _unit(vectors)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    primary = km.fit_predict(unit)
    centroids = _unit(km.cluster_centers_)

    # cosine similarity of every point to every centroid: (n, n_clusters)
    sims = unit @ centroids.T

    members: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
    for row in range(n):
        own_sim = sims[row, primary[row]]
        threshold = own_sim - overlap_margin
        for c in range(n_clusters):
            if sims[row, c] >= threshold:
                members[c].append(row)

    rng = random.Random(seed)
    oversized = 0
    for c, rows in members.items():
        if len(rows) > max_cluster_size:
            oversized += 1
            members[c] = rng.sample(rows, max_cluster_size)

    neighbors: dict[str, set[str]] = {i: set() for i in ids}
    for rows in members.values():
        member_ids = [ids[r] for r in rows]
        for a in member_ids:
            neighbors[a].update(b for b in member_ids if b != a)

    avg_neighbors = sum(len(v) for v in neighbors.values()) / n
    log.clustering_result(n_clusters, avg_neighbors, oversized, max_cluster_size)

    # Overlap membership: which datasets landed in more than one cluster
    # (the soft-boundary mechanism), and which clusters each spans.
    dataset_clusters: dict[int, list[int]] = {row: [] for row in range(n)}
    for c, rows in members.items():
        for row in rows:
            dataset_clusters[row].append(c)
    overlapping = [row for row, cs in dataset_clusters.items() if len(cs) > 1]
    max_overlap_lines = 20
    for row in overlapping[:max_overlap_lines]:
        log.cluster_overlap(ids[row], dataset_clusters[row])
    if overlapping:
        if len(overlapping) > max_overlap_lines:
            log.info(f"… and {len(overlapping) - max_overlap_lines} more overlapping dataset(s).")
        logger.info(
            "%d/%d datasets (%.1f%%) span more than one cluster.",
            len(overlapping), n, 100 * len(overlapping) / n,
        )

    return {k: list(v) for k, v in neighbors.items()}


def compute_cluster_projection(
    ids: list[str],
    vectors: np.ndarray,
    target_cluster_size: int,
    overlap_margin: float,
    max_cluster_size: int,
    seed: int,
) -> dict:
    """Cluster assignments + a 2D projection, for persistence/visualization.

    Mirrors ``build_cluster_neighbors``'s KMeans setup (same ``n_clusters``
    derivation, same soft-boundary overlap rule) so cluster ids line up with
    the neighbor-nomination clustering used during discovery — but where
    that function only returns per-dataset neighbor lists, this one returns
    the PRIMARY assignment (one cluster per dataset), the overlap-inflated
    membership lists, and a 2D coordinate per dataset (PCA on the same
    unit-normalized vectors — no new dependency, sklearn is already
    required for KMeans) so a UI can render an actual scatter plot.

    Returns:
        A JSON-serializable dict:
        ``{"n_clusters": int,
           "clusters": {cluster_id_str: [dataset_id, ...]},   # incl. overlap
           "assignments": {dataset_id: primary_cluster_id},
           "points": {dataset_id: [x, y]}}``
    """
    n = len(ids)
    if n < 2:
        return {"n_clusters": 0, "clusters": {}, "assignments": {}, "points": {}}

    n_clusters = max(1, round(n / max(1, target_cluster_size)))
    unit = _unit(vectors)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    primary = km.fit_predict(unit)
    centroids = _unit(km.cluster_centers_)
    sims = unit @ centroids.T

    overlap_members: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
    for row in range(n):
        own_sim = sims[row, primary[row]]
        threshold = own_sim - overlap_margin
        for c in range(n_clusters):
            if sims[row, c] >= threshold:
                overlap_members[c].append(row)

    rng = random.Random(seed)
    for c, rows in overlap_members.items():
        if len(rows) > max_cluster_size:
            overlap_members[c] = rng.sample(rows, max_cluster_size)

    # 2D projection for visualization only — never used for cluster
    # assignment itself, so any dimensionality-reduction quirk here can
    # never feed back into the actual (co-)clustering logic above.
    if n >= 2:
        n_components = min(2, unit.shape[1])
        coords = np.zeros((n, 2))
        reduced = PCA(n_components=n_components, random_state=seed).fit_transform(unit)
        coords[:, :n_components] = reduced
    else:
        coords = np.zeros((n, 2))

    clusters = {str(c): [ids[r] for r in rows] for c, rows in overlap_members.items()}
    assignments = {ids[row]: int(primary[row]) for row in range(n)}
    points = {ids[row]: [float(coords[row, 0]), float(coords[row, 1])] for row in range(n)}

    return {
        "n_clusters": n_clusters,
        "clusters": clusters,
        "assignments": assignments,
        "points": points,
    }


def save_cluster_projection(path: Path, projection: dict) -> None:
    """Persist ``compute_cluster_projection``'s output as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(projection, f)


class ClusterNeighborIndex:
    """Same query interface as the earlier HNSW ``NeighborIndex``, sourced
    from cluster membership (with overlap) instead of ANN search."""

    def __init__(
        self,
        ids: list[str],
        vectors: np.ndarray,
        target_cluster_size: int,
        overlap_margin: float,
        max_cluster_size: int,
        seed: int,
    ):
        self.ids = ids
        self.vectors = vectors
        self._row_of = {dataset_id: row for row, dataset_id in enumerate(ids)}
        self._neighbor_ids = build_cluster_neighbors(
            ids, vectors, target_cluster_size, overlap_margin, max_cluster_size, seed
        )

    def query_neighbors(
        self, dataset_id: str, top_k: int, cos_threshold: float
    ) -> list[tuple[str, float]]:
        """Top-k cluster-mates of a dataset above the cosine-similarity
        threshold, most similar first (same contract as the HNSW index)."""
        row = self._row_of.get(dataset_id)
        candidates = self._neighbor_ids.get(dataset_id)
        if row is None or not candidates:
            return []

        qv = self.vectors[row]
        qn = np.linalg.norm(qv) or 1.0
        scored = []
        for cand_id in candidates:
            crow = self._row_of[cand_id]
            cv = self.vectors[crow]
            cn = np.linalg.norm(cv) or 1.0
            sim = float(np.dot(qv, cv) / (qn * cn))
            if sim >= cos_threshold:
                scored.append((cand_id, sim))

        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]
