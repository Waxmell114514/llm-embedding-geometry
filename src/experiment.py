"""
Experiment module for running parameter sweeps and collecting metrics.
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import itertools

from .dataset import load_texts
from .embedder import get_embedder, NormalizationMethod
from .id_mle import estimate_id_with_ci, DistanceMetric


def run_experiment(
    model_configs: List[Dict],
    k_values: List[int],
    normalization_methods: List[NormalizationMethod],
    distance_metrics: List[DistanceMetric],
    id_methods: List[str],
    data_path: str = "data/texts.csv",
    n_bootstrap: int = 100,
    output_path: str = "outputs/metrics.csv",
    random_state: int = 42,
    use_cache: bool = True
):
    """
    Run comprehensive experiment with parameter sweep.
    
    Args:
        model_configs: List of model configurations, each a dict with:
            - model_type: "openai" or "sentence-transformer"
            - model_name: specific model name
        k_values: List of k values for kNN-MLE
        normalization_methods: List of normalization methods
        distance_metrics: List of distance metrics
        id_methods: List of ID estimation methods
        data_path: Path to text data CSV
        n_bootstrap: Number of bootstrap samples
        output_path: Path to save metrics CSV
        random_state: Random seed for reproducibility
        use_cache: Whether to use embedding cache
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load texts
    print("Loading texts...")
    texts, df_texts = load_texts(data_path)
    print(f"Loaded {len(texts)} texts")
    
    # Initialize results list
    results = []
    
    # Total number of experiments
    total_experiments = 0
    for model_config in model_configs:
        for norm in normalization_methods:
            for metric in distance_metrics:
                for method in id_methods:
                    if method == "knn-mle":
                        total_experiments += len(k_values)
                    else:
                        total_experiments += 1
    
    print(f"Running {total_experiments} experiments...")
    
    # Iterate over all parameter combinations
    pbar = tqdm(total=total_experiments, desc="Experiments")
    
    for model_config in model_configs:
        model_type = model_config["model_type"]
        model_name = model_config.get("model_name")
        
        print(f"\n{'='*60}")
        print(f"Model: {model_type} - {model_name}")
        print(f"{'='*60}")
        
        # Get embedder
        try:
            embedder = get_embedder(
                model_type=model_type,
                model_name=model_name
            )
        except Exception as e:
            print(f"Error creating embedder: {e}")
            print(f"Skipping model: {model_type} - {model_name}")
            pbar.update(
                len(normalization_methods) * len(distance_metrics) *
                (sum(len(k_values) if m == "knn-mle" else 1 for m in id_methods))
            )
            continue
        
        # Get embeddings for each normalization method
        embeddings_dict = {}
        for norm in normalization_methods:
            print(f"  Getting embeddings with normalization: {norm}")
            try:
                embeddings = embedder.embed(
                    texts,
                    normalization=norm,
                    use_cache=use_cache
                )
                embeddings_dict[norm] = embeddings
                print(f"    Shape: {embeddings.shape}")
            except Exception as e:
                print(f"    Error: {e}")
                embeddings_dict[norm] = None
        
        # Run ID estimation experiments
        for norm in normalization_methods:
            if embeddings_dict[norm] is None:
                pbar.update(
                    len(distance_metrics) *
                    sum(len(k_values) if m == "knn-mle" else 1 for m in id_methods)
                )
                continue
                
            embeddings = embeddings_dict[norm]
            
            for metric in distance_metrics:
                for method in id_methods:
                    if method == "knn-mle":
                        k_list = k_values
                    else:
                        k_list = [None]
                    
                    for k in k_list:
                        try:
                            # Estimate ID
                            result = estimate_id_with_ci(
                                embeddings,
                                method=method,
                                k=k,
                                metric=metric,
                                n_bootstrap=n_bootstrap,
                                random_state=random_state
                            )
                            
                            # Add metadata
                            result.update({
                                "model_type": model_type,
                                "model_name": model_name or "default",
                                "normalization": norm,
                                "n_samples": embeddings.shape[0],
                                "embedding_dim": embeddings.shape[1]
                            })
                            
                            results.append(result)
                            
                        except Exception as e:
                            print(f"    Error in experiment: {e}")
                            print(f"      Config: {model_type}, {model_name}, {norm}, {metric}, {method}, k={k}")
                        
                        pbar.update(1)
    
    pbar.close()
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Reorder columns
    column_order = [
        "model_type", "model_name", "normalization", "metric", "method", "k",
        "id_estimate", "ci_lower", "ci_upper", "std",
        "n_samples", "embedding_dim", "n_bootstrap", "n_valid_bootstrap"
    ]
    df_results = df_results[[c for c in column_order if c in df_results.columns]]
    
    # Save results
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Total experiments: {len(df_results)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for model_type in df_results["model_type"].unique():
        print(f"\n{model_type}:")
        model_data = df_results[df_results["model_type"] == model_type]
        
        for norm in model_data["normalization"].unique():
            norm_data = model_data[model_data["normalization"] == norm]
            mean_id = norm_data["id_estimate"].mean()
            std_id = norm_data["id_estimate"].std()
            print(f"  {norm}: ID = {mean_id:.2f} ± {std_id:.2f}")
    
    return df_results


def run_default_experiment(
    use_openai: bool = False,
    output_dir: str = "outputs"
):
    """
    Run experiment with default parameters.
    
    Args:
        use_openai: Whether to include OpenAI models (requires API key)
        output_dir: Directory to save outputs
    """
    # Model configurations
    model_configs = [
        {"model_type": "sentence-transformer", "model_name": "BAAI/bge-small-en-v1.5"},
    ]
    
    if use_openai:
        model_configs.append({
            "model_type": "openai",
            "model_name": "text-embedding-3-small"
        })
    
    # Parameter ranges
    k_values = [5, 10, 15, 20, 30, 40, 50]
    normalization_methods = ["none", "l2", "center"]
    distance_metrics = ["euclidean", "cosine"]
    id_methods = ["knn-mle", "twonn"]
    
    # Run experiment
    output_path = os.path.join(output_dir, "metrics.csv")
    
    df_results = run_experiment(
        model_configs=model_configs,
        k_values=k_values,
        normalization_methods=normalization_methods,
        distance_metrics=distance_metrics,
        id_methods=id_methods,
        output_path=output_path,
        n_bootstrap=100,
        random_state=42
    )
    
    return df_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ID estimation experiments")
    parser.add_argument("--use-openai", action="store_true", help="Include OpenAI models")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    print("Starting experiments...")
    print(f"OpenAI models: {'Yes' if args.use_openai else 'No'}")
    
    df = run_default_experiment(
        use_openai=args.use_openai,
        output_dir=args.output_dir
    )
    
    print("\nDone!")
