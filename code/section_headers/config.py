from pathlib import Path

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "collected_patents.jsonl"
MAPPING = ROOT / "data" / "header_groups.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 42
MIN_DOCS = 20

SENTENCES_PER_CHUNK = 10
CHUNK_OVERLAP = 1

CLUSTERS = {
    "model": "BAAI/bge-small-en-v1.5",
    "n_contexts": 20,
    "context_chars": 300,
    "w_name": 1.0,
    "w_context": 1.0,
    "w_position": 1.0,
    "method": "agglomerative",
    "k_range": (6, 30),
    "n_clusters": 12,
    "umap_dim": 5,
    "umap_neighbors": 15,
    "min_samples": 3,
    "eps": 0.0,
    "max_noise": 0.3,
    "min_clusters": 6,
}
