"""
LLM Embedding Geometry Analysis Package
"""
from .dataset import load_texts, get_texts_by_concept, get_texts_by_template, generate_sample_data
from .embedder import get_embedder, OpenAIEmbedder, SentenceTransformerEmbedder, normalize_embeddings
from .id_mle import knn_mle_levina_bickel, twonn, estimate_id_with_ci, bootstrap_id_estimate
from .experiment import run_experiment, run_default_experiment
from .plot import plot_id_vs_k, plot_heatmap, plot_comparison, generate_all_plots

__version__ = "0.1.0"

__all__ = [
    # Dataset
    "load_texts",
    "get_texts_by_concept",
    "get_texts_by_template",
    "generate_sample_data",
    # Embedder
    "get_embedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "normalize_embeddings",
    # ID estimation
    "knn_mle_levina_bickel",
    "twonn",
    "estimate_id_with_ci",
    "bootstrap_id_estimate",
    # Experiment
    "run_experiment",
    "run_default_experiment",
    # Plotting
    "plot_id_vs_k",
    "plot_heatmap",
    "plot_comparison",
    "generate_all_plots",
]
