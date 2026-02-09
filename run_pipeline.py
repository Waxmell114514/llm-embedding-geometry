#!/usr/bin/env python
"""
Main script to run the complete pipeline for LLM embedding geometry analysis.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiment import run_default_experiment
from plot import generate_all_plots


def main():
    """Run the complete pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run LLM embedding geometry analysis pipeline"
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Include OpenAI models (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip experiments and only generate plots from existing metrics"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("LLM EMBEDDING GEOMETRY ANALYSIS")
    print("="*70)
    print()
    
    if not args.skip_experiments:
        print("STEP 1: Running experiments...")
        print("-"*70)
        
        try:
            df = run_default_experiment(
                use_openai=args.use_openai,
                output_dir=args.output_dir
            )
            print("\n✓ Experiments completed successfully")
        except Exception as e:
            print(f"\n✗ Error running experiments: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Skipping experiments (using existing metrics)")
    
    print("\n" + "="*70)
    print("STEP 2: Generating visualizations...")
    print("-"*70)
    
    try:
        metrics_path = os.path.join(args.output_dir, "metrics.csv")
        generate_all_plots(
            metrics_path=metrics_path,
            output_dir=args.output_dir
        )
        print("\n✓ Visualizations generated successfully")
    except Exception as e:
        print(f"\n✗ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - metrics.csv: Detailed experiment results")
    print(f"  - id_vs_k.png: ID estimates vs k parameter")
    print(f"  - id_heatmap.png: Heatmap of ID estimates")
    print(f"  - comparison.png: Comparison across models and methods")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
