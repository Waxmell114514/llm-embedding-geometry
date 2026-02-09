"""
Visualization module for plotting ID estimation results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, List


def plot_id_vs_k(
    df: pd.DataFrame,
    output_path: str = "outputs/id_vs_k.png",
    figsize: tuple = (12, 8)
):
    """
    Plot intrinsic dimension vs k for different configurations.
    
    Args:
        df: DataFrame with experiment results
        output_path: Path to save figure
        figsize: Figure size
    """
    # Filter for kNN-MLE method only
    df_knn = df[df["method"] == "knn-mle"].copy()
    
    if len(df_knn) == 0:
        print("No kNN-MLE results to plot")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot configurations
    plot_idx = 0
    
    for model_type in df_knn["model_type"].unique():
        if plot_idx >= 4:
            break
            
        ax = axes[plot_idx]
        
        # Filter data for this model
        model_data = df_knn[df_knn["model_type"] == model_type]
        
        # Plot each configuration
        for (norm, metric), group in model_data.groupby(["normalization", "metric"]):
            # Sort by k
            group = group.sort_values("k")
            
            label = f"{norm}, {metric}"
            ax.plot(group["k"], group["id_estimate"], marker='o', label=label, alpha=0.7)
            
            # Add confidence intervals
            ax.fill_between(
                group["k"],
                group["ci_lower"],
                group["ci_upper"],
                alpha=0.2
            )
        
        # Set labels and title
        model_name = model_data["model_name"].iloc[0]
        ax.set_xlabel("k (number of neighbors)", fontsize=11)
        ax.set_ylabel("Intrinsic Dimension", fontsize=11)
        ax.set_title(f"{model_type}: {model_name}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_heatmap(
    df: pd.DataFrame,
    output_path: str = "outputs/id_heatmap.png",
    figsize: tuple = (14, 10)
):
    """
    Plot heatmap of ID estimates across different configurations.
    
    Args:
        df: DataFrame with experiment results
        output_path: Path to save figure
        figsize: Figure size
    """
    # Get unique models
    models = df["model_type"].unique()
    n_models = len(models)
    
    fig, axes = plt.subplots(n_models, 2, figsize=figsize)
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, model_type in enumerate(models):
        model_data = df[df["model_type"] == model_type]
        model_name = model_data["model_name"].iloc[0]
        
        # Heatmap for kNN-MLE
        ax = axes[idx, 0]
        knn_data = model_data[model_data["method"] == "knn-mle"]
        
        if len(knn_data) > 0:
            # Create pivot table
            pivot = knn_data.pivot_table(
                values="id_estimate",
                index=["normalization", "metric"],
                columns="k",
                aggfunc="mean"
            )
            
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                ax=ax,
                cbar_kws={'label': 'ID Estimate'}
            )
            ax.set_title(f"{model_type}: {model_name} - kNN-MLE", fontweight='bold')
            ax.set_xlabel("k")
            ax.set_ylabel("Configuration")
        else:
            ax.text(0.5, 0.5, "No kNN-MLE data", ha='center', va='center')
            ax.set_title(f"{model_type}: {model_name} - kNN-MLE", fontweight='bold')
        
        # Heatmap for TwoNN
        ax = axes[idx, 1]
        twonn_data = model_data[model_data["method"] == "twonn"]
        
        if len(twonn_data) > 0:
            # Create pivot table
            pivot = twonn_data.pivot_table(
                values="id_estimate",
                index=["normalization", "metric"],
                aggfunc="mean"
            )
            
            # Reshape for heatmap
            pivot_reshaped = pivot.to_frame().T
            
            sns.heatmap(
                pivot_reshaped,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                ax=ax,
                cbar_kws={'label': 'ID Estimate'}
            )
            ax.set_title(f"{model_type}: {model_name} - TwoNN", fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel("")
        else:
            ax.text(0.5, 0.5, "No TwoNN data", ha='center', va='center')
            ax.set_title(f"{model_type}: {model_name} - TwoNN", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_comparison(
    df: pd.DataFrame,
    output_path: str = "outputs/comparison.png",
    figsize: tuple = (14, 6)
):
    """
    Plot comparison of ID estimates across models and methods.
    
    Args:
        df: DataFrame with experiment results
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Comparison by normalization method
    ax = axes[0]
    
    # Group by model and normalization
    grouped = df.groupby(["model_type", "model_name", "normalization"]).agg({
        "id_estimate": "mean",
        "std": "mean"
    }).reset_index()
    
    # Create labels
    grouped["label"] = grouped["model_type"] + "\n" + grouped["model_name"]
    
    # Plot
    x_pos = np.arange(len(grouped["label"].unique()))
    width = 0.25
    
    norms = grouped["normalization"].unique()
    for i, norm in enumerate(norms):
        norm_data = grouped[grouped["normalization"] == norm]
        positions = x_pos + (i - len(norms)/2 + 0.5) * width
        
        ax.bar(
            positions,
            norm_data["id_estimate"],
            width,
            label=norm,
            alpha=0.8
        )
    
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Mean Intrinsic Dimension", fontsize=11)
    ax.set_title("ID Estimates by Normalization Method", fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped["label"].unique(), fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Comparison by method
    ax = axes[1]
    
    # Group by model and method
    grouped = df.groupby(["model_type", "model_name", "method"]).agg({
        "id_estimate": "mean",
        "std": "mean"
    }).reset_index()
    
    # Create labels
    grouped["label"] = grouped["model_type"] + "\n" + grouped["model_name"]
    
    # Plot
    x_pos = np.arange(len(grouped["label"].unique()))
    width = 0.35
    
    methods = grouped["method"].unique()
    for i, method in enumerate(methods):
        method_data = grouped[grouped["method"] == method]
        positions = x_pos + (i - len(methods)/2 + 0.5) * width
        
        ax.bar(
            positions,
            method_data["id_estimate"],
            width,
            label=method,
            alpha=0.8
        )
    
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Mean Intrinsic Dimension", fontsize=11)
    ax.set_title("ID Estimates by Method", fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped["label"].unique(), fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def generate_all_plots(
    metrics_path: str = "outputs/metrics.csv",
    output_dir: str = "outputs"
):
    """
    Generate all plots from metrics CSV.
    
    Args:
        metrics_path: Path to metrics CSV file
        output_dir: Directory to save plots
    """
    # Load metrics
    print(f"Loading metrics from {metrics_path}")
    df = pd.read_csv(metrics_path)
    print(f"Loaded {len(df)} results")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_id_vs_k(df, output_path=f"{output_dir}/id_vs_k.png")
    plot_heatmap(df, output_path=f"{output_dir}/id_heatmap.png")
    plot_comparison(df, output_path=f"{output_dir}/comparison.png")
    
    print("\nAll plots generated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots from metrics")
    parser.add_argument("--metrics", default="outputs/metrics.csv", help="Path to metrics CSV")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    generate_all_plots(
        metrics_path=args.metrics,
        output_dir=args.output_dir
    )
