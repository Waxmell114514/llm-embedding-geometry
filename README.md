# LLM Embedding Geometry Analysis

A minimal viable research repository for analyzing the geometric structure of text embedding models through intrinsic dimension estimation.

## Overview

This project investigates the intrinsic geometric properties of semantic spaces produced by mainstream text embedding models (OpenAI text-embedding-3-small, BGE, GTE, etc.) using rigorous statistical methods.

### Key Features

- **Multiple Embedding Models**: Unified interface for OpenAI and open-source models (Sentence Transformers)
- **Intrinsic Dimension Estimation**: Implementation of kNN-MLE (Levina–Bickel) and TwoNN methods
- **Comprehensive Analysis**: Parameter sensitivity analysis across k, normalization strategies, and distance metrics
- **Statistical Rigor**: Bootstrap confidence intervals for all estimates
- **Full Reproducibility**: Public dataset, cached embeddings, and deterministic experiments

## Research Background

### What is Intrinsic Dimension?

The intrinsic dimension (ID) of a dataset represents the minimum number of parameters needed to describe the data, regardless of the ambient space dimensionality. For example, points on a 2D surface embedded in 3D space have an intrinsic dimension of 2.

### Why Does It Matter for LLM Embeddings?

Understanding the intrinsic dimension of embedding spaces helps us:

1. **Understand representation capacity**: Lower ID suggests embeddings compress semantic information into fewer effective dimensions
2. **Optimize downstream tasks**: Knowing the true dimensionality can guide dimension reduction strategies
3. **Compare models**: Different embedding models may organize semantic information differently
4. **Detect redundancy**: High ambient dimension with low intrinsic dimension indicates redundant features

### Methods

**kNN-MLE (Levina & Bickel, 2004)**: Estimates ID by analyzing the distribution of distances to k-nearest neighbors. For each point, it computes the maximum likelihood estimate based on distance ratios.

**TwoNN (Facco et al., 2017)**: Uses the ratio of distances to the first and second nearest neighbors. Based on the theory that this ratio follows a specific distribution in d-dimensional spaces.

## Dataset

The analysis uses a structured dataset of 900 texts:
- **150 concepts**: Abstract concepts, emotions, objects, actions, etc.
- **6 templates**: Different sentence structures to reduce template bias
- **Total**: 150 × 6 = 900 text samples

Examples:
- "This is about love."
- "The concept of happiness is important."
- "Understanding freedom requires careful thought."

## Installation

### Requirements

- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Waxmell114514/llm-embedding-geometry.git
cd llm-embedding-geometry

# Install dependencies
pip install -r requirements.txt

# (Optional) Set up OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Quick Start

### Run with Open-Source Models Only

```bash
python run_pipeline.py
```

This will:
1. Load the 900-text dataset
2. Generate embeddings using BGE-small-en-v1.5
3. Estimate intrinsic dimensions with various parameters
4. Generate visualizations

### Run with OpenAI Models

```bash
# Make sure you have set OPENAI_API_KEY in .env
python run_pipeline.py --use-openai
```

### Generate Plots from Existing Results

```bash
python run_pipeline.py --skip-experiments
```

### Test Installation

```bash
python test_installation.py
```

## Project Structure

```
llm-embedding-geometry/
├── data/
│   └── texts.csv              # 900 texts (150 concepts × 6 templates)
├── src/
│   ├── __init__.py            # Package initialization
│   ├── dataset.py             # Data loading and generation
│   ├── embedder.py            # Unified embedding interface
│   ├── id_mle.py              # ID estimation algorithms
│   ├── experiment.py          # Parameter sweep experiments
│   └── plot.py                # Visualization
├── outputs/                    # Results and plots (generated)
│   ├── metrics.csv
│   ├── id_vs_k.png
│   ├── id_heatmap.png
│   └── comparison.png
├── embeddings_cache/          # Cached embeddings (generated)
├── run_pipeline.py            # Main pipeline script
├── demo.py                    # Demo with simulated embeddings
├── test_installation.py       # Installation verification script
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore
├── CONTRIBUTING.md            # Contribution guidelines
├── EXAMPLES.md                # Detailed usage examples
├── LICENSE
└── README.md
```

## Documentation

- **README.md** (this file): Overview, installation, and quick start
- **[EXAMPLES.md](EXAMPLES.md)**: Detailed usage examples and tutorials
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guidelines for contributing to the project

## Usage Examples

### Using Individual Modules

```python
# Load dataset
from src.dataset import load_texts
texts, df = load_texts("data/texts.csv")

# Get embeddings
from src.embedder import get_embedder
embedder = get_embedder("sentence-transformer", "BAAI/bge-small-en-v1.5")
embeddings = embedder.embed(texts, normalization="l2")

# Estimate intrinsic dimension
from src.id_mle import estimate_id_with_ci
result = estimate_id_with_ci(embeddings, method="knn-mle", k=10)
print(f"ID: {result['id_estimate']:.2f} (95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}])")
```

### Run Custom Experiments

```python
from src.experiment import run_experiment

model_configs = [
    {"model_type": "sentence-transformer", "model_name": "BAAI/bge-small-en-v1.5"}
]

run_experiment(
    model_configs=model_configs,
    k_values=[5, 10, 20, 30],
    normalization_methods=["l2", "none"],
    distance_metrics=["euclidean", "cosine"],
    id_methods=["knn-mle", "twonn"],
    output_path="outputs/custom_metrics.csv"
)
```

## Key Results

### Main Findings

(To be updated after running experiments)

**Expected observations based on literature:**

1. **Intrinsic dimension varies significantly** across different embedding models
2. **Normalization affects ID estimates**: L2 normalization typically reduces estimated ID
3. **ID estimates stabilize** as k increases in kNN-MLE
4. **Distance metric matters**: Euclidean vs cosine distance can yield different estimates
5. **Bootstrap confidence intervals** help assess estimation reliability

### Parameter Sensitivity

The project systematically investigates:
- **k values**: [5, 10, 15, 20, 30, 40, 50] for kNN-MLE
- **Normalization**: none, L2, centering
- **Distance metrics**: Euclidean, cosine
- **Methods**: kNN-MLE, TwoNN

## Open Questions

1. **What is the "true" intrinsic dimension of semantic spaces?**
   - Do different estimation methods converge?
   - How do we validate ID estimates for high-dimensional embeddings?

2. **How does model architecture affect embedding geometry?**
   - Do larger models create higher or lower dimensional spaces?
   - Do instruction-tuned models differ from base models?

3. **Does intrinsic dimension predict downstream performance?**
   - Is there an optimal ID for specific tasks?
   - Can we use ID as a model selection criterion?

4. **How stable are ID estimates across different datasets?**
   - Domain-specific vs general text
   - Different languages and cultures

5. **What is the relationship between ID and semantic structure?**
   - How does ID relate to concept hierarchies?
   - Does topic diversity affect ID estimates?

## Contributing

This is a research repository. Contributions are welcome:
- Additional embedding models
- New ID estimation methods
- Improved visualizations
- Extended datasets
- Bug fixes and optimizations

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llm_embedding_geometry,
  title = {LLM Embedding Geometry Analysis},
  author = {LIU, Ziwei},
  year = {2026},
  url = {https://github.com/Waxmell114514/llm-embedding-geometry}
}
```

### Key References

- Levina, E., & Bickel, P. (2004). Maximum likelihood estimation of intrinsic dimension. *Advances in Neural Information Processing Systems*, 17.
- Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017). Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific Reports*, 7(1), 12140.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact & Collaboration

### Cold Email Template for Industry Professionals

```
Subject: Research on Intrinsic Dimension of Text Embeddings - Collaboration Opportunity

Dear [Name],

I am reaching out to share research on the geometric structure of text embedding 
spaces that may be relevant to [Company]'s work on [specific application].

I have developed an open-source analysis framework that estimates the intrinsic 
dimension of embedding models (OpenAI, BGE, GTE, etc.) using rigorous statistical 
methods (kNN-MLE, TwoNN). The findings suggest that:

1. [Key finding 1 - to be filled after experiments]
2. [Key finding 2 - to be filled after experiments]
3. [Key finding 3 - to be filled after experiments]

These insights could potentially inform:
- Model selection and optimization for [specific use case]
- Dimension reduction strategies
- Understanding representation capacity trade-offs

The complete codebase and results are publicly available at:
https://github.com/Waxmell114514/llm-embedding-geometry

I would welcome the opportunity to discuss how these findings might apply to 
your work, or to collaborate on extending this research.

Best regards,
[Your Name]
[Your Affiliation]
[Contact Information]
```

---

**Note**: This is a minimal viable research repository focused on reproducibility and clarity. For production use, consider additional optimizations, error handling, and scalability improvements.