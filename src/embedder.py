"""
Unified embedding interface for OpenAI and open-source models.
Supports caching and various normalization strategies.
"""
import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Optional, Literal
import numpy as np
from abc import ABC, abstractmethod


NormalizationMethod = Literal["none", "l2", "center", "whiten"]


class BaseEmbedder(ABC):
    """Base class for embedding models."""
    
    def __init__(self, cache_dir: str = "embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    @abstractmethod
    def _get_embeddings_uncached(self, texts: List[str]) -> np.ndarray:
        """Get embeddings without caching (to be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name for caching purposes."""
        pass
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key from texts."""
        # Create a hash of all texts
        text_str = "|||".join(texts)
        text_hash = hashlib.md5(text_str.encode()).hexdigest()
        return f"{self.get_model_name()}_{text_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Get embeddings with caching support.
        
        Args:
            texts: List of text strings
            use_cache: Whether to use cache
            
        Returns:
            embeddings: numpy array of shape (n_texts, embedding_dim)
        """
        if use_cache:
            cache_key = self._get_cache_key(texts)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Get embeddings
        embeddings = self._get_embeddings_uncached(texts)
        
        # Cache if requested
        if use_cache:
            cache_key = self._get_cache_key(texts)
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
        
        return embeddings
    
    def embed(
        self,
        texts: List[str],
        normalization: NormalizationMethod = "none",
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Get embeddings with normalization.
        
        Args:
            texts: List of text strings
            normalization: Normalization method ("none", "l2", "center", "whiten")
            use_cache: Whether to use cache
            
        Returns:
            embeddings: numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.get_embeddings(texts, use_cache=use_cache)
        embeddings = normalize_embeddings(embeddings, method=normalization)
        return embeddings


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding model."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        cache_dir: str = "embeddings_cache"
    ):
        super().__init__(cache_dir)
        self.model = model
        
        # Get API key
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Import OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def get_model_name(self) -> str:
        return f"openai_{self.model}"
    
    def _get_embeddings_uncached(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API."""
        from tqdm import tqdm
        
        # Process in batches to handle rate limits
        batch_size = 100
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Getting {self.model} embeddings"):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformer (open-source) embedding model."""
    
    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str = "embeddings_cache",
        device: Optional[str] = None
    ):
        super().__init__(cache_dir)
        self.model_name = model
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model, device=device)
    
    def get_model_name(self) -> str:
        return f"st_{self.model_name.replace('/', '_')}"
    
    def _get_embeddings_uncached(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from Sentence Transformer model."""
        from tqdm import tqdm
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        return embeddings


def normalize_embeddings(
    embeddings: np.ndarray,
    method: NormalizationMethod = "none"
) -> np.ndarray:
    """
    Normalize embeddings using specified method.
    
    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        method: Normalization method
            - "none": No normalization
            - "l2": L2 normalization (unit vectors)
            - "center": Center to zero mean
            - "whiten": Whiten (zero mean, unit variance)
            
    Returns:
        Normalized embeddings
    """
    if method == "none":
        return embeddings
    
    # Use appropriate epsilon based on dtype
    eps = np.finfo(embeddings.dtype).eps * 10
    
    if method == "l2":
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + eps)
    
    elif method == "center":
        # Center to zero mean
        mean = np.mean(embeddings, axis=0, keepdims=True)
        return embeddings - mean
    
    elif method == "whiten":
        # Whiten: zero mean and unit variance
        mean = np.mean(embeddings, axis=0, keepdims=True)
        std = np.std(embeddings, axis=0, keepdims=True)
        return (embeddings - mean) / (std + eps)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_embedder(
    model_type: str,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseEmbedder:
    """
    Factory function to get embedder instance.
    
    Args:
        model_type: Type of model ("openai" or "sentence-transformer")
        model_name: Specific model name (optional)
        **kwargs: Additional arguments for embedder
        
    Returns:
        Embedder instance
    """
    if model_type == "openai":
        model = model_name or "text-embedding-3-small"
        return OpenAIEmbedder(model=model, **kwargs)
    
    elif model_type == "sentence-transformer":
        model = model_name or "BAAI/bge-small-en-v1.5"
        return SentenceTransformerEmbedder(model=model, **kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test embedder
    texts = ["Hello, world!", "This is a test.", "Embeddings are useful."]
    
    print("Testing Sentence Transformer embedder...")
    try:
        embedder = get_embedder("sentence-transformer")
        embeddings = embedder.embed(texts, normalization="l2")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"L2 norms: {np.linalg.norm(embeddings, axis=1)}")
    except Exception as e:
        print(f"Error: {e}")
