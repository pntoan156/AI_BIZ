"""
Module cung cấp embedding function
"""
import os
import io
import aiohttp
from typing import Dict, Any, Optional, List, Union
from embeddings.aicore_embedding import AiCoreEmbedding
from dotenv import load_dotenv
from util.env_loader import get_env

load_dotenv()

def get_embedding_model(provider: str = None, model_params: Optional[Dict[str, Any]] = None):
    """
    Lấy embedding model dựa trên provider và tham số
    
    Args:
        provider (str, optional): Provider của embedding model ("openai", "google", "huggingface", "aicore"). Defaults to None.
        model_params (Optional[Dict[str, Any]], optional): Tham số cho model. Defaults to None.
        
    Returns:
        Embeddings: Embedding model được khởi tạo
    """
    if model_params is None:
        model_params = {}
    
    # Nếu không chỉ định provider, lấy từ biến môi trường EMBEDDING_PROVIDER
    if provider is None:
        provider = get_env("EMBEDDING_PROVIDER", "aicore").lower()
    
    if provider == "aicore":
        return _get_aicore_embeddings(model_params)
    else:
        raise ValueError(f"Provider không được hỗ trợ: {provider}")

def _get_aicore_embeddings(model_params: Dict[str, Any]) -> AiCoreEmbedding:
    """
    Lấy AiCore embedding model
    
    Args:
        model_params (Dict[str, Any]): Tham số cho model
        
    Returns:
        AiCoreEmbedding: AiCore embedding model
    """
    default_params = {
        "api_url": get_env("EMBEDDING_API_URL", "https://aiservice.abc.vn/jina/embedding"),
        "api_key": get_env("EMBEDDING_API_KEY", ""),
        "truncate_dim": 768
    }
    
    # Merge default params và model params
    params = {**default_params, **model_params}
    
    return AiCoreEmbedding(**params)

# --- Placeholder Embedding for Image Store Initialization ---

class PlaceholderImageEmbedding:
    """
    Một lớp embedding giả lập chỉ dùng để khởi tạo VectorStore
    khi vector thực tế được tính toán trước.
    Nó cần phải callable và trả về một list/array để _get_vector_dim hoạt động.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        # Tạo sẵn vector 0 để trả về
        self._dummy_vector = [0.0] * dimension

    def __call__(self, text: str) -> List[float]:
        """Làm cho đối tượng callable, trả về vector 0."""
        # print(f"PlaceholderImageEmbedding called with text: '{text}'") # Debugging
        return self._dummy_vector

    def embed_query(self, text: str) -> List[float]:
        """Triển khai phương thức embed_query theo interface Embeddings."""
        return self._dummy_vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Triển khai phương thức embed_documents theo interface Embeddings."""
        return [self._dummy_vector for _ in texts]