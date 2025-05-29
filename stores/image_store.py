import os
import time
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from stores.vector_store_provider import get_vector_store
from embeddings.embedding_provider import get_embedding_model
from stores.base_vector_store import BaseVectorStore
from util.env_loader import get_env

class ImageStore(BaseVectorStore):
    def __init__(self, collection_name: str = "image_collection", recreate_collection: bool = False):
        """
        Khởi tạo ImageVectorStore
        
        Args:
            collection_name (str): Tên collection
            recreate_collection (bool): Có tạo lại collection không
        """
        self.collection_name = collection_name
        self.embedding_function = get_embedding_model("aicore")
        self.vectorstore = get_vector_store(
            self.embedding_function, 
            collection_name, 
            recreate_collection=recreate_collection
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 2000,  # Tăng batch size
        **kwargs
    ) -> List[str]:
        """
        Thêm văn bản vào store với batch processing tối ưu (loại bỏ trùng lặp)
        
        Args:
            texts: Danh sách văn bản
            metadatas: Danh sách metadata tương ứng
            batch_size: Kích thước mỗi batch để insert
            **kwargs: Các tham số bổ sung
            
        Returns:
            List[str]: Danh sách ID của các văn bản đã thêm
        """
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        try:
            print(f"Bắt đầu xử lý {len(texts)} items với batch_size={batch_size}")
            
            # Sử dụng phương thức tối ưu từ vectorstore thay vì insert trực tiếp
            # Điều này sẽ sử dụng schema đã được tối ưu và tránh trùng lặp dữ liệu
            ids = self.vectorstore.bulk_insert_texts(
                texts=texts,
                metadatas=metadatas,
                batch_size=batch_size,
                **kwargs
            )
            
            print(f"Hoàn thành! Đã xử lý {len(ids)} items")
            return ids
            
        except Exception as e:
            print(f"Lỗi khi thêm texts: {str(e)}")
            if "422" in str(e):
                print("Lỗi API embedding - kiểm tra lại URL và API key")
            raise

    def delete(self, ids: List[str]) -> None:
        """
        Xóa các bản ghi theo ID
        
        Args:
            ids: Danh sách ID cần xóa
        """
        expr = f'id in {ids}'
        self.vectorstore.collection.delete(expr)

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
        try:
            # Thực hiện full-text search
            expr = f'text like "%{query}%"'
            
            results = self.vectorstore.collection.query(
                expr=expr,
                output_fields=["id", "text"],
                limit=k
            )
            
            # Tính toán điểm số đơn giản dựa trên số lần xuất hiện của từ khóa
            docs_with_scores = []
            query_words = query.lower().split()
            
            for item in results:
                text = item.get("text", "").lower()
                score = 0
                for word in query_words:
                    score += text.count(word)
                
                # Chuẩn hóa điểm số
                if len(query_words) > 0:
                    score = score / len(query_words)
                
                doc = {
                    "id": item.get("id"),
                    "text": item.get("text"),
                }
                docs_with_scores.append((doc, min(score, 1.0)))
            
            # Sắp xếp kết quả theo điểm số giảm dần
            docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            return docs_with_scores[:k]
            
        except Exception as e:
            print(f"Lỗi khi thực hiện fulltext search: {e}")
            return []

    def hybrid_search(
        self, 
        query: str, 
        k: int = 4, 
        alpha: float = 0.5,
        category: str = "all",
        app_name: str = "all",
        **kwargs
    ) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm văn bản kết hợp giữa vector similarity và full-text search
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            alpha: Trọng số cho kết quả vector similarity (0-1)
            category: Danh mục để lọc kết quả
            app_name: Tên app để lọc kết quả
            
        Returns:
            List[Tuple[Any, float]]: Danh sách tuple gồm tài liệu và điểm số kết hợp
        """
        # Lấy kết quả từ cả hai phương pháp
        vector_results = self.similarity_search(query, k=k*2, category=category, app_name=app_name, **kwargs)
        fulltext_results = self.fulltext_search(query, k=k*2, **kwargs)
        
        # Tạo map từ ID đến kết quả và điểm số
        results_map = {}
        
        # Thêm kết quả vector search với trọng số alpha
        for doc, score in vector_results:
            doc_id = doc["id"]
            if doc_id not in results_map:
                results_map[doc_id] = {"doc": doc, "vector_score": 0, "text_score": 0}
            results_map[doc_id]["vector_score"] = score
        
        # Thêm kết quả fulltext search với trọng số (1-alpha)
        for doc, score in fulltext_results:
            doc_id = doc["id"]
            if doc_id not in results_map:
                results_map[doc_id] = {"doc": doc, "vector_score": 0, "text_score": 0}
            results_map[doc_id]["text_score"] = score
        
        print(f"fulltext_results: {fulltext_results}")
        print(f"vector_results: {vector_results}")
        
        # Tính điểm kết hợp
        hybrid_results = []
        for doc_id, result in results_map.items():
            hybrid_score = alpha * result["vector_score"] + (1 - alpha) * result["text_score"]
            hybrid_results.append((result["doc"], hybrid_score))
        
        # Sắp xếp theo điểm kết hợp giảm dần
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_results[:k]

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        category: str = "all",
        app_name: str = "all"
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Tìm kiếm similarity dựa trên text query (sử dụng phương thức tối ưu)
        
        Args:
            query: Text query để tìm kiếm
            k: Số lượng kết quả trả về
            category: Danh mục để lọc kết quả
            app_name: Tên app để lọc kết quả
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: Danh sách kết quả và score
        """
        try:
            # Sử dụng phương thức similarity_search tối ưu từ vectorstore
            results = self.vectorstore.similarity_search(query, k=k)
            
            # Filter theo category và app_name nếu cần
            filtered_results = []
            for doc, score in results:
                metadata = doc.get("metadata", {})
                
                # Kiểm tra điều kiện lọc
                if category != "all" and metadata.get("category") != category:
                    continue
                if app_name != "all" and metadata.get("app_name") != app_name:
                    continue
                
                # Format lại kết quả để tương thích
                formatted_doc = {
                    "id": doc.get("id"),
                    "text": doc.get("text"),
                    "image_path": metadata.get("image_path", ""),
                    "category": metadata.get("category", ""),
                    "style": metadata.get("style", ""),
                    "app_name": metadata.get("app_name", ""),
                    "metadata": metadata
                }
                
                filtered_results.append((formatted_doc, score))
            
            return filtered_results[:k]
            
        except Exception as e:
            print(f"Lỗi khi thực hiện similarity search: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về collection
        
        Returns:
            Dict[str, Any]: Thông tin collection
        """
        try:
            collection = self.vectorstore.collection
            
            # Lấy số lượng entities
            collection.load()
            num_entities = collection.num_entities
            
            # Lấy schema info
            schema_info = {
                "collection_name": collection.name,
                "num_entities": num_entities,
                "fields": [field.name for field in collection.schema.fields],
                "description": collection.description
            }
            
            return schema_info
            
        except Exception as e:
            print(f"Lỗi khi lấy thông tin collection: {e}")
            return {
                "collection_name": self.collection_name,
                "num_entities": 0,
                "error": str(e)
            }