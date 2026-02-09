# Usage Examples

This document provides detailed usage examples for the LLM Embedding Geometry Analysis toolkit.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Working with Different Models](#working-with-different-models)
3. [Custom Experiments](#custom-experiments)
4. [Advanced Analysis](#advanced-analysis)
5. [Interpreting Results](#interpreting-results)

## Basic Usage

### Quick Start with Demo Data

The fastest way to test the pipeline is with the demo script:

```bash
python demo.py
```

This generates simulated embeddings and runs the complete analysis pipeline.

### Running with Real Models

To run with actual embedding models:

```bash
# With open-source models only (no API key needed)
python run_pipeline.py

# With OpenAI models (requires OPENAI_API_KEY in .env)
python run_pipeline.py --use-openai

# Skip experiments and just regenerate plots
python run_pipeline.py --skip-experiments
```

## Working with Different Models

### Using Sentence Transformers

```python
from src.embedder import get_embedder
from src.dataset import load_texts

# Load data
texts, df = load_texts("data/texts.csv")

# Get embedder
embedder = get_embedder(
    model_type="sentence-transformer",
    model_name="BAAI/bge-small-en-v1.5"
)

# Generate embeddings
embeddings = embedder.embed(texts, normalization="l2")
print(f"Embeddings shape: {embeddings.shape}")
```

### Using OpenAI Models

```python
from src.embedder import get_embedder
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Get embedder
embedder = get_embedder(
    model_type="openai",
    model_name="text-embedding-3-small"
)

# Generate embeddings (with caching)
embeddings = embedder.embed(texts, normalization="l2", use_cache=True)
```

### Trying Different Models

```python
models = [
    ("sentence-transformer", "BAAI/bge-small-en-v1.5"),
    ("sentence-transformer", "sentence-transformers/all-MiniLM-L6-v2"),
    ("sentence-transformer", "thenlper/gte-small"),
]

for model_type, model_name in models:
    embedder = get_embedder(model_type, model_name)
    embeddings = embedder.embed(texts[:100], normalization="l2")
    print(f"{model_name}: {embeddings.shape}")
```

## Custom Experiments

### Single ID Estimation

```python
from src.id_mle import estimate_id_with_ci
import numpy as np

# Generate or load embeddings
embeddings = np.random.randn(500, 384)  # Example

# Estimate ID with kNN-MLE
result = estimate_id_with_ci(
    embeddings,
    method="knn-mle",
    k=10,
    metric="euclidean",
    n_bootstrap=100,
    random_state=42
)

print(f"ID: {result['id_estimate']:.2f}")
print(f"95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
print(f"Std: {result['std']:.2f}")
```

### Comparing Different k Values

```python
from src.id_mle import knn_mle_levina_bickel

k_values = [5, 10, 15, 20, 30, 40, 50]
id_estimates = []

for k in k_values:
    id_est = knn_mle_levina_bickel(embeddings, k=k, metric="euclidean")
    id_estimates.append(id_est)
    print(f"k={k}: ID={id_est:.2f}")

# Plot
import matplotlib.pyplot as plt
plt.plot(k_values, id_estimates, marker='o')
plt.xlabel('k')
plt.ylabel('Intrinsic Dimension')
plt.title('ID vs k')
plt.savefig('id_vs_k_custom.png')
```

### Custom Parameter Sweep

```python
from src.experiment import run_experiment

# Define models to test
model_configs = [
    {"model_type": "sentence-transformer", "model_name": "BAAI/bge-small-en-v1.5"},
    {"model_type": "sentence-transformer", "model_name": "sentence-transformers/all-MiniLM-L6-v2"},
]

# Run custom experiment
df_results = run_experiment(
    model_configs=model_configs,
    k_values=[10, 20, 30],
    normalization_methods=["l2", "none"],
    distance_metrics=["euclidean", "cosine"],
    id_methods=["knn-mle", "twonn"],
    output_path="outputs/custom_metrics.csv",
    n_bootstrap=50
)
```

## Advanced Analysis

### Analyzing Specific Concepts

```python
from src.dataset import load_texts, get_texts_by_concept

texts, df = load_texts("data/texts.csv")

# Get texts for specific concepts
love_texts = get_texts_by_concept(df, "love")
freedom_texts = get_texts_by_concept(df, "freedom")

# Compare their embeddings
embedder = get_embedder("sentence-transformer", "BAAI/bge-small-en-v1.5")
love_embeddings = embedder.embed(love_texts, normalization="l2")
freedom_embeddings = embedder.embed(freedom_texts, normalization="l2")

# Estimate ID for each
from src.id_mle import estimate_id_with_ci
love_id = estimate_id_with_ci(love_embeddings, method="twonn")
freedom_id = estimate_id_with_ci(freedom_embeddings, method="twonn")

print(f"Love ID: {love_id['id_estimate']:.2f}")
print(f"Freedom ID: {freedom_id['id_estimate']:.2f}")
```

### Comparing Normalization Strategies

```python
normalizations = ["none", "l2", "center", "whiten"]
results = {}

for norm in normalizations:
    embeddings_norm = embedder.embed(texts, normalization=norm, use_cache=False)
    result = estimate_id_with_ci(embeddings_norm, method="knn-mle", k=20)
    results[norm] = result['id_estimate']
    print(f"{norm}: {result['id_estimate']:.2f}")
```

### Analyzing Distance Metrics

```python
from src.id_mle import knn_mle_levina_bickel

embeddings = embedder.embed(texts, normalization="l2")

for metric in ["euclidean", "cosine", "manhattan"]:
    try:
        id_est = knn_mle_levina_bickel(embeddings, k=20, metric=metric)
        print(f"{metric}: {id_est:.2f}")
    except Exception as e:
        print(f"{metric}: Error - {e}")
```

## Interpreting Results

### Understanding ID Estimates

- **Low ID (< 10)**: The embeddings lie on a low-dimensional manifold
- **Medium ID (10-50)**: Moderate dimensional structure
- **High ID (> 50)**: Nearly fills the ambient space

### Confidence Intervals

Wider confidence intervals suggest:
- Less stable estimates
- Sensitivity to sampling
- Need for more data or different k

### k Value Selection

- Small k (5-10): Captures local structure, more variance
- Medium k (15-30): Balance of local/global, generally stable
- Large k (40+): Global structure, may underestimate ID

### Normalization Effects

- **L2 normalization**: Often reduces ID, emphasizes angular relationships
- **Centering**: Removes global offset, can affect ID estimates
- **No normalization**: Preserves original structure, may be affected by scaling

### Method Comparison

- **kNN-MLE**: Stable for various k, good for general use
- **TwoNN**: Simple, fast, good for quick estimates

## Batch Processing

For large-scale experiments:

```python
import pandas as pd

# Process multiple datasets
datasets = ["data/texts.csv", "data/other_texts.csv"]
all_results = []

for dataset_path in datasets:
    texts, df = load_texts(dataset_path)
    embeddings = embedder.embed(texts, normalization="l2")
    result = estimate_id_with_ci(embeddings, method="knn-mle", k=20)
    result['dataset'] = dataset_path
    all_results.append(result)

results_df = pd.DataFrame(all_results)
results_df.to_csv("batch_results.csv", index=False)
```

## Troubleshooting

### Out of Memory

If you encounter memory issues with large datasets:

```python
# Process in batches
batch_size = 100
all_embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    batch_embeddings = embedder.embed(batch, use_cache=True)
    all_embeddings.append(batch_embeddings)

embeddings = np.vstack(all_embeddings)
```

### Cache Management

```python
# Clear cache
import shutil
shutil.rmtree("embeddings_cache", ignore_errors=True)

# Disable cache
embeddings = embedder.embed(texts, use_cache=False)
```

## Further Reading

- See [README.md](README.md) for overview and installation
- See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Check the [issues](https://github.com/Waxmell114514/llm-embedding-geometry/issues) for known problems and discussions
