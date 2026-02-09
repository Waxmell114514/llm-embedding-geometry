#!/usr/bin/env python
"""
Demo script that runs a complete analysis with simulated embeddings.
This allows testing the full pipeline without requiring API keys or network access.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import load_texts
from id_mle import estimate_id_with_ci
from plot import generate_all_plots


def simulate_embeddings(texts, embedding_dim=384, intrinsic_dim=None, random_state=42):
    """
    Create simulated embeddings for testing purposes.
    
    Args:
        texts: List of text strings
        embedding_dim: Dimension of embedding space
        intrinsic_dim: Intrinsic dimension (if None, uses random embeddings)
        random_state: Random seed
        
    Returns:
        Simulated embeddings
    """
    np.random.seed(random_state)
    n_texts = len(texts)
    
    if intrinsic_dim is not None:
        # Create embeddings with specific intrinsic dimension
        X_intrinsic = np.random.randn(n_texts, intrinsic_dim)
        projection = np.random.randn(intrinsic_dim, embedding_dim)
        embeddings = X_intrinsic @ projection
    else:
        # Random embeddings
        embeddings = np.random.randn(n_texts, embedding_dim)
    
    # Add some structure based on concepts
    # This is a simplified simulation - real embeddings have much richer structure
    return embeddings


def normalize_embeddings(embeddings, method="none"):
    """Normalize embeddings."""
    if method == "none":
        return embeddings
    
    # Use appropriate epsilon based on dtype
    eps = np.finfo(embeddings.dtype).eps * 10
    
    if method == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + eps)
    elif method == "center":
        mean = np.mean(embeddings, axis=0, keepdims=True)
        return embeddings - mean
    return embeddings


def run_demo():
    """Run a complete demo with simulated embeddings."""
    print("="*70)
    print("LLM EMBEDDING GEOMETRY ANALYSIS - DEMO MODE")
    print("="*70)
    print("\nThis demo uses simulated embeddings for testing purposes.")
    print("For real analysis, use run_pipeline.py with actual embedding models.\n")
    
    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load texts
    print("Step 1: Loading dataset...")
    texts, df_texts = load_texts("data/texts.csv")
    print(f"  Loaded {len(texts)} texts")
    
    # Simulate embeddings for different "models"
    print("\nStep 2: Generating simulated embeddings...")
    
    models = {
        "simulated-low-dim": {"dim": 384, "intrinsic_dim": 8},
        "simulated-high-dim": {"dim": 768, "intrinsic_dim": 20},
    }
    
    results = []
    
    for model_name, config in models.items():
        print(f"\n  Model: {model_name}")
        
        # Generate embeddings
        embeddings_base = simulate_embeddings(
            texts,
            embedding_dim=config["dim"],
            intrinsic_dim=config["intrinsic_dim"],
            random_state=42
        )
        
        # Test different normalizations
        for norm_method in ["none", "l2", "center"]:
            print(f"    Normalization: {norm_method}")
            embeddings = normalize_embeddings(embeddings_base, norm_method)
            
            # Test different metrics
            for metric in ["euclidean", "cosine"]:
                # kNN-MLE with different k values
                for k in [5, 10, 15, 20, 30, 40, 50]:
                    result = estimate_id_with_ci(
                        embeddings,
                        method="knn-mle",
                        k=k,
                        metric=metric,
                        n_bootstrap=50,
                        random_state=42
                    )
                    
                    result.update({
                        "model_type": "simulated",
                        "model_name": model_name,
                        "normalization": norm_method,
                        "n_samples": len(texts),
                        "embedding_dim": config["dim"]
                    })
                    results.append(result)
                
                # TwoNN
                result = estimate_id_with_ci(
                    embeddings,
                    method="twonn",
                    metric=metric,
                    n_bootstrap=50,
                    random_state=42
                )
                
                result.update({
                    "model_type": "simulated",
                    "model_name": model_name,
                    "normalization": norm_method,
                    "n_samples": len(texts),
                    "embedding_dim": config["dim"]
                })
                results.append(result)
    
    # Save results
    print("\nStep 3: Saving results...")
    df_results = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        "model_type", "model_name", "normalization", "metric", "method", "k",
        "id_estimate", "ci_lower", "ci_upper", "std",
        "n_samples", "embedding_dim", "n_bootstrap", "n_valid_bootstrap"
    ]
    df_results = df_results[[c for c in column_order if c in df_results.columns]]
    
    metrics_path = os.path.join(output_dir, "metrics.csv")
    df_results.to_csv(metrics_path, index=False)
    print(f"  Saved to {metrics_path}")
    
    # Print summary
    print("\nStep 4: Summary statistics...")
    for model_name in df_results["model_name"].unique():
        model_data = df_results[df_results["model_name"] == model_name]
        mean_id = model_data["id_estimate"].mean()
        std_id = model_data["id_estimate"].std()
        print(f"  {model_name}: ID = {mean_id:.2f} ± {std_id:.2f}")
    
    # Generate plots
    print("\nStep 5: Generating visualizations...")
    try:
        generate_all_plots(
            metrics_path=metrics_path,
            output_dir=output_dir
        )
        print("  ✓ All visualizations generated")
    except Exception as e:
        print(f"  Warning: Could not generate all plots: {e}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETED")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - metrics.csv: Detailed experiment results")
    print(f"  - id_vs_k.png: ID estimates vs k parameter")
    print(f"  - id_heatmap.png: Heatmap of ID estimates")
    print(f"  - comparison.png: Comparison across models and methods")
    print("\nNote: These results use simulated embeddings for demonstration.")
    print("For real analysis, use: python run_pipeline.py")
    print()


if __name__ == "__main__":
    run_demo()
