"""
Module định nghĩa lớp cơ sở cho các loại vector store
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple

class BaseVectorStore(ABC):
    """
    Lớp cơ sở cho các loại vector store. Định nghĩa các phương thức cơ bản mà mọi 
    vector store cần thực hiện.
    """
    
    @abstractmethod
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Thêm văn bản và metadata vào vector store
        
        Args:
            texts: Danh sách các văn bản cần thêm
            metadatas: Danh sách metadata tương ứng với mỗi văn bản
            
        Returns:
            List[str]: Danh sách ID của các văn bản đã thêm
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """
        Thêm tài liệu vào vector store
        
        Args:
            documents: Danh sách tài liệu cần thêm
            
        Returns:
            List[str]: Danh sách ID của các tài liệu đã thêm
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs
    ) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm văn bản tương tự với query dựa trên độ tương đồng về vector
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            
        Returns:
            List[Tuple[Any, float]]: Danh sách tuple gồm tài liệu và độ tương đồng
        """
        pass
    
    @abstractmethod
    def fulltext_search(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs
    ) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm văn bản dựa trên full-text search
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            
        Returns:
            List[Tuple[Any, float]]: Danh sách tuple gồm tài liệu và điểm số
        """
        pass
    
    @abstractmethod
    def hybrid_search(
        self, 
        query: str, 
        k: int = 4, 
        alpha: float = 0.5, 
        **kwargs
    ) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm văn bản kết hợp giữa vector similarity và full-text search
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            alpha: Trọng số cho kết quả vector similarity (0-1),
                  Giá trị alpha càng cao thì kết quả sẽ thiên về vector similarity hơn,
                  ngược lại giá trị alpha thấp sẽ thiên về full-text search
            
        Returns:
            List[Tuple[Any, float]]: Danh sách tuple gồm tài liệu và điểm số kết hợp
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str], **kwargs) -> None:
        """
        Xóa các văn bản khỏi vector store dựa trên ID
        
        Args:
            ids: Danh sách ID cần xóa
        """
        pass
    