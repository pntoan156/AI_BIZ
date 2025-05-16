"""
Module chứa AiCoreEmbedding class
"""
import os
import json
import requests
from typing import List, Optional

from langchain_core.embeddings import Embeddings

class AiCoreEmbedding(Embeddings):
    """
    Class wrapper cho MISA Ai Core Embedding API
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        truncate_dim: int = 768,
        **kwargs
    ):
        """
        Khởi tạo AiCoreEmbedding
        
        Args:
            api_url (Optional[str], optional): URL của API embedding. Defaults to None.
            api_key (Optional[str], optional): API key. Defaults to None.
            truncate_dim (int, optional): Số chiều embedding. Defaults to 768.
        """
        self.api_url = api_url or os.getenv("EMBEDDING_API_URL", "https://aiservice.abc.vn/jina/embedding")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
        self.truncate_dim = truncate_dim
        self.headers = {"x-api-key": self.api_key} if self.api_key else {}
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed danh sách văn bản
        
        Args:
            texts (List[str]): Danh sách văn bản cần embed
            
        Returns:
            List[List[float]]: Danh sách vector embedding
        """
        payload = json.dumps({
            "texts": texts,
            "task": "retrieval.document",
            "truncate_dim": self.truncate_dim
        })
        
        response = requests.request("POST", self.api_url, headers=self.headers, data=payload)
        response.raise_for_status()
        
        return response.json()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed câu truy vấn
        
        Args:
            text (str): Câu truy vấn cần embed
            
        Returns:
            List[float]: Vector embedding
        """
        payload = json.dumps({
            "texts": [text],
            "task": "retrieval.query",
            "truncate_dim": self.truncate_dim
        })
        
        response = requests.request("POST", self.api_url, headers=self.headers, data=payload)
        response.raise_for_status()
        
        # API trả về list các embedding, lấy embedding đầu tiên
        return response.json()[0]
    
    def __call__(self, text: str) -> List[float]:
        """
        Cho phép gọi instance như một hàm để tương thích với các vector store
        
        Args:
            text (str): Văn bản cần embed
            
        Returns:
            List[float]: Vector embedding
        """
        # Xử lý text đơn lẻ bằng cách gọi embed_query
        return self.embed_query(text) 