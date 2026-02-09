# Contributing to LLM Embedding Geometry

Thank you for your interest in contributing to this research project!

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:
1. Check if the issue already exists in the GitHub issue tracker
2. If not, create a new issue with:
   - Clear description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, etc.)

### Contributing Code

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Make your changes following the code style guidelines
4. Add tests if applicable
5. Update documentation as needed
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature-name`
8. Create a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex algorithms

### Areas for Contribution

**New Features:**
- Additional embedding models (e.g., GTE, E5, Instructor)
- New ID estimation methods (e.g., PCA-based, MiND-ML)
- Advanced normalization strategies
- Interactive visualizations
- Performance optimizations

**Improvements:**
- Better error handling
- More comprehensive tests
- Improved documentation
- Additional example datasets
- Benchmark comparisons

**Research Extensions:**
- Domain-specific analysis
- Multi-language embeddings
- Temporal analysis of embedding spaces
- Correlation with downstream task performance

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llm-embedding-geometry.git
cd llm-embedding-geometry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python demo.py
```

## Testing

Before submitting a PR:
1. Run the demo to ensure basic functionality: `python demo.py`
2. Test with real models if possible: `python run_pipeline.py`
3. Verify all visualizations are generated
4. Check that your code doesn't break existing functionality

## Documentation

When adding new features:
- Update the README.md with usage examples
- Add docstrings to new functions/classes
- Update the module documentation if needed
- Include references for new algorithms

## Questions?

Feel free to open an issue for discussion or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
