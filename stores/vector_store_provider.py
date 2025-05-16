"""
Module cung cấp vector store dựa trên cấu hình
"""
import os
from typing import Any, Dict, List, Optional, Union
from util.env_loader import get_env

from stores.base_vector_store import BaseVectorStore
from stores.milvus_vector_store import MilvusVectorStore

def get_vector_store(
    embedding_function: Any,
    collection_name: str,
    provider: str = None,
    **kwargs
) -> BaseVectorStore:
    """
    Lấy vector store phù hợp dựa trên provider được chỉ định.
    
    Args:
        collection_name: Tên của collection
        embedding_function: Hàm embedding để chuyển đổi văn bản thành vector
        provider: Loại vector store provider ('milvus')
                  Mặc định sẽ lấy từ biến môi trường VECTOR_STORE_PROVIDER
        **kwargs: Các tham số bổ sung cho vector store
        
    Returns:
        BaseVectorStore: Vector store được khởi tạo
        
    Raises:
        ValueError: Nếu provider không được hỗ trợ
    """
    # Lấy provider từ biến môi trường nếu không được chỉ định
    if not provider:
        provider = get_env("VECTOR_STORE_PROVIDER", "milvus").lower()
    
    if provider == "milvus":
        return MilvusVectorStore(
            collection_name=collection_name,
            embedding_function=embedding_function,
            **kwargs
        )
    else:
        raise ValueError(f"Provider không được hỗ trợ: {provider}. Chỉ hỗ trợ 'milvus'") 