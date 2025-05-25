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