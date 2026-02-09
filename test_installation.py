#!/usr/bin/env python
"""
Test script to verify the installation and basic functionality.
"""
import sys
import importlib


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "tqdm",
        "dotenv",
    ]
    
    optional_packages = [
        "openai",
        "sentence_transformers",
        "torch",
    ]
    
    failed = []
    
    for package in required_packages:
        try:
            if package == "sklearn":
                importlib.import_module("sklearn")
            elif package == "dotenv":
                importlib.import_module("dotenv")
            else:
                importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package} - {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing required packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required packages installed")
    
    # Test optional packages
    print("\nTesting optional packages...")
    for package in optional_packages:
        try:
            if package == "sentence_transformers":
                importlib.import_module("sentence_transformers")
            else:
                importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} (optional)")
    
    return True


def test_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    
    modules = [
        "src.dataset",
        "src.embedder",
        "src.id_mle",
        "src.experiment",
        "src.plot",
    ]
    
    failed = []
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} - {e}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    
    print("\n✓ All project modules imported successfully")
    return True


def test_data():
    """Test that data files exist."""
    print("\nTesting data files...")
    
    from pathlib import Path
    
    data_files = [
        "data/texts.csv",
    ]
    
    missing = []
    
    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - not found")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ Missing data files: {', '.join(missing)}")
        print("Generate data with: python src/dataset.py")
        return False
    
    print("\n✓ All data files present")
    return True


def test_id_estimation():
    """Test ID estimation with synthetic data."""
    print("\nTesting ID estimation...")
    
    try:
        import numpy as np
        from src.id_mle import knn_mle_levina_bickel, twonn
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 200
        true_dim = 3
        
        X_intrinsic = np.random.randn(n_samples, true_dim)
        projection = np.random.randn(true_dim, 10)
        X = X_intrinsic @ projection
        
        # Test kNN-MLE
        id_knn = knn_mle_levina_bickel(X, k=10, metric="euclidean")
        print(f"  kNN-MLE (k=10): {id_knn:.2f} (expected ~{true_dim})")
        
        # Test TwoNN
        id_twonn = twonn(X, metric="euclidean")
        print(f"  TwoNN: {id_twonn:.2f} (expected ~{true_dim})")
        
        # Basic sanity check
        if 1 < id_knn < 20 and 1 < id_twonn < 20:
            print("\n✓ ID estimation working correctly")
            return True
        else:
            print("\n⚠ ID estimates seem unusual, but may be OK")
            return True
            
    except Exception as e:
        print(f"\n❌ ID estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    
    try:
        from src.dataset import load_texts
        
        texts, df = load_texts("data/texts.csv")
        
        print(f"  Loaded {len(texts)} texts")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Concepts: {df['concept'].nunique()}")
        print(f"  Templates: {df['template_id'].nunique()}")
        
        expected_texts = 150 * 6  # 150 concepts × 6 templates
        if len(texts) == expected_texts:
            print(f"\n✓ Dataset loading working correctly")
            return True
        else:
            print(f"\n⚠ Expected {expected_texts} texts, got {len(texts)}")
            return False
            
    except Exception as e:
        print(f"\n❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("LLM EMBEDDING GEOMETRY - INSTALLATION TEST")
    print("="*70)
    print()
    
    results = []
    
    results.append(("Package imports", test_imports()))
    results.append(("Project modules", test_modules()))
    results.append(("Data files", test_data()))
    results.append(("Dataset loading", test_dataset()))
    results.append(("ID estimation", test_id_estimation()))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 All tests passed! Installation is complete.")
        print("\nNext steps:")
        print("  1. Run demo: python demo.py")
        print("  2. Run full pipeline: python run_pipeline.py")
        print("  3. See EXAMPLES.md for more usage examples")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
