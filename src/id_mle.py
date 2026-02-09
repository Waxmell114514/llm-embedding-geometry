"""
Intrinsic dimension estimation using kNN-MLE and TwoNN methods.
"""
import numpy as np
from typing import Tuple, Optional, Literal
from sklearn.neighbors import NearestNeighbors


DistanceMetric = Literal["euclidean", "cosine"]


def compute_knn_distances(
    X: np.ndarray,
    k: int,
    metric: DistanceMetric = "euclidean"
) -> np.ndarray:
    """
    Compute k-nearest neighbor distances.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        k: Number of nearest neighbors
        metric: Distance metric to use
        
    Returns:
        distances: Array of shape (n_samples, k) with distances to k-NNs
    """
    # Use sklearn's NearestNeighbors
    # k+1 because the first neighbor is the point itself
    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Remove the first column (distance to self, which is 0)
    return distances[:, 1:]


def knn_mle_levina_bickel(
    X: np.ndarray,
    k: int,
    metric: DistanceMetric = "euclidean"
) -> float:
    """
    Estimate intrinsic dimension using kNN-MLE (Levina & Bickel, 2004).
    
    Reference:
    Levina, E., & Bickel, P. (2004). Maximum likelihood estimation of intrinsic dimension.
    Advances in Neural Information Processing Systems, 17.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        k: Number of nearest neighbors
        metric: Distance metric to use
        
    Returns:
        Estimated intrinsic dimension
    """
    n_samples = X.shape[0]
    
    # Get k-NN distances
    distances = compute_knn_distances(X, k, metric)
    
    # Use appropriate epsilon based on dtype
    eps = np.finfo(X.dtype).eps * 10
    
    # Compute the MLE estimate for each point
    # ID = (k-1) / sum(log(r_k / r_i)) for i=1..k-1
    # where r_i is distance to i-th nearest neighbor
    
    estimates = []
    for i in range(n_samples):
        # Get distances for this point
        r = distances[i, :]  # shape (k,)
        
        # Distance to k-th neighbor
        r_k = r[-1]
        
        # Avoid division by zero or log(0)
        if r_k < eps:
            continue
            
        # Distances to first k-1 neighbors
        r_j = r[:-1]  # shape (k-1,)
        
        # Avoid division by zero
        valid = r_j > eps
        if not np.any(valid):
            continue
        
        # Compute log ratio sum
        log_ratio_sum = np.sum(np.log(r_k / r_j[valid]))
        
        if log_ratio_sum > 0:
            id_estimate = len(r_j[valid]) / log_ratio_sum
            estimates.append(id_estimate)
    
    if len(estimates) == 0:
        return np.nan
    
    # Return mean estimate
    return np.mean(estimates)


def twonn(
    X: np.ndarray,
    metric: DistanceMetric = "euclidean"
) -> float:
    """
    Estimate intrinsic dimension using TwoNN method (Facco et al., 2017).
    
    Reference:
    Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017).
    Estimating the intrinsic dimension of datasets by a minimal neighborhood information.
    Scientific Reports, 7(1), 12140.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        metric: Distance metric to use
        
    Returns:
        Estimated intrinsic dimension
    """
    # Get distances to first and second nearest neighbors
    distances = compute_knn_distances(X, k=2, metric=metric)
    
    # r1: distance to first NN
    # r2: distance to second NN
    r1 = distances[:, 0]
    r2 = distances[:, 1]
    
    # Use appropriate epsilon based on dtype
    eps = np.finfo(X.dtype).eps * 10
    
    # Compute mu = r2 / r1
    valid = (r1 > eps) & (r2 > eps)
    mu = r2[valid] / r1[valid]
    
    if len(mu) == 0:
        return np.nan
    
    # The empirical CDF of mu follows F(mu) = mu^d for d-dimensional data
    # We estimate d using the fact that E[log(mu)] = 1/d
    # So d = 1 / E[log(mu)]
    
    log_mu = np.log(mu)
    mean_log_mu = np.mean(log_mu)
    
    if mean_log_mu <= 0:
        return np.nan
    
    id_estimate = 1.0 / mean_log_mu
    return id_estimate


def bootstrap_id_estimate(
    X: np.ndarray,
    estimator_fn,
    n_bootstrap: int = 100,
    sample_ratio: float = 0.8,
    random_state: Optional[int] = None,
    **estimator_kwargs
) -> Tuple[float, float, float, np.ndarray]:
    """
    Estimate intrinsic dimension with bootstrap confidence intervals.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        estimator_fn: Function to estimate ID (e.g., knn_mle_levina_bickel or twonn)
        n_bootstrap: Number of bootstrap samples
        sample_ratio: Ratio of samples to use in each bootstrap
        random_state: Random seed for reproducibility
        **estimator_kwargs: Additional arguments for estimator function
        
    Returns:
        mean_estimate: Mean ID estimate across bootstrap samples
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        estimates: Array of all bootstrap estimates
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_sample = int(n_samples * sample_ratio)
    
    estimates = []
    for i in range(n_bootstrap):
        # Random sample with replacement
        indices = np.random.choice(n_samples, size=n_sample, replace=True)
        X_sample = X[indices]
        
        # Estimate ID
        id_estimate = estimator_fn(X_sample, **estimator_kwargs)
        
        if not np.isnan(id_estimate):
            estimates.append(id_estimate)
    
    estimates = np.array(estimates)
    
    if len(estimates) == 0:
        return np.nan, np.nan, np.nan, estimates
    
    # Compute statistics
    mean_estimate = np.mean(estimates)
    ci_lower = np.percentile(estimates, 2.5)
    ci_upper = np.percentile(estimates, 97.5)
    
    return mean_estimate, ci_lower, ci_upper, estimates


def estimate_id_with_ci(
    X: np.ndarray,
    method: str = "knn-mle",
    k: Optional[int] = None,
    metric: DistanceMetric = "euclidean",
    n_bootstrap: int = 100,
    random_state: Optional[int] = None
) -> dict:
    """
    Estimate intrinsic dimension with confidence intervals.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        method: Method to use ("knn-mle" or "twonn")
        k: Number of neighbors for knn-mle (if None, uses heuristic)
        metric: Distance metric
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
        
    Returns:
        Dictionary with estimation results
    """
    # Set default k if not provided
    if k is None and method == "knn-mle":
        # Heuristic: k ≈ log(n)
        k = max(5, min(50, int(np.log(X.shape[0]))))
    
    # Select estimator
    if method == "knn-mle":
        estimator_fn = knn_mle_levina_bickel
        estimator_kwargs = {"k": k, "metric": metric}
    elif method == "twonn":
        estimator_fn = twonn
        estimator_kwargs = {"metric": metric}
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Estimate with bootstrap
    mean_id, ci_lower, ci_upper, estimates = bootstrap_id_estimate(
        X,
        estimator_fn,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        **estimator_kwargs
    )
    
    return {
        "method": method,
        "k": k if method == "knn-mle" else None,
        "metric": metric,
        "id_estimate": mean_id,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": np.std(estimates) if len(estimates) > 0 else np.nan,
        "n_bootstrap": n_bootstrap,
        "n_valid_bootstrap": len(estimates)
    }


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing ID estimators with synthetic data...")
    
    # Generate 3D data embedded in 10D space
    np.random.seed(42)
    n_samples = 500
    true_dim = 3
    
    # Generate data on a 3D manifold
    X_intrinsic = np.random.randn(n_samples, true_dim)
    
    # Embed in higher dimensional space
    embed_dim = 10
    projection = np.random.randn(true_dim, embed_dim)
    X = X_intrinsic @ projection
    
    print(f"Data shape: {X.shape}")
    print(f"True intrinsic dimension: {true_dim}")
    
    # Test kNN-MLE
    for k in [5, 10, 20]:
        id_est = knn_mle_levina_bickel(X, k=k, metric="euclidean")
        print(f"kNN-MLE (k={k}): {id_est:.2f}")
    
    # Test TwoNN
    id_est = twonn(X, metric="euclidean")
    print(f"TwoNN: {id_est:.2f}")
    
    # Test with bootstrap
    print("\nTesting with bootstrap confidence intervals...")
    result = estimate_id_with_ci(X, method="knn-mle", k=10, n_bootstrap=50)
    print(f"kNN-MLE: {result['id_estimate']:.2f} (95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}])")
    
    result = estimate_id_with_ci(X, method="twonn", n_bootstrap=50)
    print(f"TwoNN: {result['id_estimate']:.2f} (95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}])")
