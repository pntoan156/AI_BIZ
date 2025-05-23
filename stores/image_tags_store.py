import os
from typing import List, Dict, Any, Optional, Tuple
from stores.vector_store_provider import get_vector_store
from embeddings.embedding_provider import get_embedding_model
from stores.base_vector_store import BaseVectorStore

class ImageTagsStore(BaseVectorStore):
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
        **kwargs
    ) -> List[str]:
        """
        Thêm văn bản vào store
        
        Args:
            texts: Danh sách văn bản
            metadatas: Danh sách metadata tương ứng
            **kwargs: Các tham số bổ sung
            
        Returns:
            List[str]: Danh sách ID của các văn bản đã thêm
        """
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        try:
            # Tạo vector từ tất cả texts cùng lúc
            vectors = [self.embedding_function(text) for text in texts]
            
            # Chuẩn bị dữ liệu để chèn
            data = []
            for i, (text, metadata, vector) in enumerate(zip(texts, metadatas, vectors)):
                # Tạo ID nếu không có
                image_id = metadata.get("id", str(i))
                
                # Chuẩn bị dữ liệu cho một bản ghi
                record = {
                    "id": image_id,
                    "vector": vector,
                    "text": text,
                    "image_path": metadata.get("image_path", ""),
                    "metadata": metadata
                }
                data.append(record)
            
            # Chèn dữ liệu vào collection
            insert_result = self.vectorstore.collection.insert(data)
            self.vectorstore.collection.flush()
            
            return [record["id"] for record in data]
            
        except Exception as e:
            print(f"Lỗi khi thêm texts: {str(e)}")
            if "422" in str(e):
                print("Lỗi API embedding - kiểm tra lại URL và API key")
            raise

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """
        Thêm tài liệu vào store
        
        Args:
            documents: Danh sách tài liệu
            **kwargs: Các tham số bổ sung
            
        Returns:
            List[str]: Danh sách ID của các tài liệu đã thêm
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

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
        limit: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Tìm kiếm hybrid (kết hợp vector và fulltext)
        
        Args:
            query: Câu truy vấn
            limit: Số lượng kết quả
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: Danh sách kết quả và score
        """
        # Lấy kết quả từ vector search
        vector_results = self.similarity_search(query, limit)
        
        # Lấy kết quả từ fulltext search
        text_results = self.fulltext_search(query, limit)
        
        # Kết hợp và sắp xếp kết quả
        combined_results = {}
        for doc, score in vector_results + text_results:
            doc_id = doc["id"]
            if doc_id not in combined_results or score > combined_results[doc_id][1]:
                combined_results[doc_id] = (doc, score)
                
        return sorted(combined_results.values(), key=lambda x: x[1], reverse=True)[:limit]

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê của collection
        
        Returns:
            Dict[str, Any]: Thông tin thống kê
        """
        stats = self.vectorstore.collection.get_stats()
        return {
            "total_rows": stats.get("row_count", 0),
            "collection_name": self.collection_name
        }
        
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
            alpha: Trọng số cho kết quả vector similarity (0-1)
            
        Returns:
            List[Tuple[Any, float]]: Danh sách tuple gồm tài liệu và điểm số kết hợp
        """
        # Lấy kết quả từ cả hai phương pháp
        vector_results = self.similarity_search(query, k=k*2, **kwargs)
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
        k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Tìm kiếm similarity dựa trên text query
        
        Args:
            query: Text query để tìm kiếm
            limit: Số lượng kết quả trả về
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: Danh sách kết quả và score
        """
        # Tạo vector từ query
        query_vector = self.embedding_function.embed_query(query)
            
        # Thực hiện tìm kiếm
        results = self.vectorstore.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
            output_fields=["id", "text", "image_path"]
        )
        
        # Format kết quả
        docs_with_scores = []
        for hit in results[0]:
            doc = {
                "id": hit.entity.get("id"),
                "text": hit.entity.get("text"),
                "image_path": hit.entity.get("image_path")
            }
            docs_with_scores.append((doc, hit.score))
            
        return docs_with_scores
    
async def embed_and_store_images(images_data: List[Dict], recreate_collection: bool = False):
    """
    Embed và lưu trữ ảnh vào vector store
    
    Args:
        images_data (List[Dict]): Danh sách dữ liệu ảnh
        recreate_collection (bool): Có tạo lại collection không
        
    Returns:
        Dict: Kết quả xử lý
    """
    # Khởi tạo image vector store
    image_store = ImageTagsStore(recreate_collection=recreate_collection)
    
    try:
        # Chuẩn bị texts và metadata
        texts = []
        metadatas = []
        
        for img_data in images_data:
            texts.append(img_data['image_name'])
            metadatas.append({
                "id": img_data['image_id'],
                "image_path": img_data['image_path'],
                "image_name": img_data['image_name']
            })
        
        # Thêm vào vector store
        image_store.add_texts(texts, metadatas)
        
        return {
            "success": True,
            "processed_count": len(images_data),
            "error_count": 0,
            "total": len(images_data)
        }
        
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "processed_count": 0,
            "error_count": len(images_data),
            "total": len(images_data)
        } 