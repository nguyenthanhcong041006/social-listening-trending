from umap import UMAP
from hdbscan import HDBSCAN
from typing import Dict, Any

def create_umap_model(config: Dict[str, Any] = None) -> UMAP:
    """
    Reduces embedding dimensionality using UMAP (Uniform Manifold Approximation and Projection)
    in preparation for density-based clustering.
    """
    cfg = config or {}
    return UMAP(
        n_neighbors=cfg.get("n_neighbors", 15),
        n_components=cfg.get("n_components", 5),
        min_dist=cfg.get("min_dist", 0.0),
        metric=cfg.get("metric", "cosine"),
        random_state=42
    )

def create_hdbscan_model(config: Dict[str, Any] = None) -> HDBSCAN:
    """
    Step 2: Clustering
    Hierarchical density-based spatial clustering using HDBSCAN.
    Automatically identifies topics while isolating noise outliers (label -1).
    """
    cfg = config or {}
    return HDBSCAN(
        min_cluster_size=cfg.get("min_cluster_size", 10),
        min_samples=cfg.get("min_samples", 5),
        metric=cfg.get("metric", "euclidean"),
        prediction_data=cfg.get("prediction_data", True)
    )
