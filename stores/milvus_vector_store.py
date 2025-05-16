"""
Module triển khai Milvus vector store
"""
import os
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema
from pymilvus import DataType, MilvusException
from util.env_loader import get_env

from stores.base_vector_store import BaseVectorStore

class MilvusVectorStore(BaseVectorStore):
    """
    Triển khai vector store sử dụng Milvus
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_function: Any,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        text_field: str = "text", # Field for text content
        # vector_field is removed, specific fields below
        id_field: str = "id",
        database_id_field: str = "database_id",
        metadata_fields: Optional[List[Tuple[str, str]]] = None, # e.g., [("metadata", DataType.JSON)]
        clip_dimension: int = 512, # Default CLIP dimension
        resnet_dimension: int = 2048, # Default ResNet dimension
        text_vector_field: str = "vector", # Field name for text embeddings
        recreate_collection: bool = False,
        **kwargs
    ):
        """
        Khởi tạo Milvus vector store
        
        Args:
            collection_name: Tên của collection
            embedding_function: Hàm embedding (primarily for text). Can be None if only adding precomputed vectors.
            uri: URI kết nối tới Milvus server. Mặc định lấy từ biến môi trường MILVUS_URI
            db_name: Tên database. Mặc định lấy từ biến môi trường MILVUS_DB_NAME
            user: Tên người dùng. Mặc định lấy từ biến môi trường MILVUS_USER
            password: Mật khẩu. Mặc định lấy từ biến môi trường MILVUS_PASSWORD
            text_field: Tên trường chứa văn bản gốc.
            id_field: Tên trường chứa ID chính (primary key).
            database_id_field: Tên trường chứa DatabaseId (partition key).
            metadata_fields: Danh sách các trường metadata bổ sung và kiểu dữ liệu (e.g., [("metadata", DataType.JSON)]).
            clip_dimension: Số chiều cho vector CLIP.
            resnet_dimension: Số chiều cho vector ResNet.
            text_vector_field: Tên trường để lưu vector embedding của text.
            recreate_collection: Có tạo lại collection nếu đã tồn tại không
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function # Used for text embedding and potentially dim inference
        self.text_field = text_field
        self.id_field = id_field
        self.database_id_field = database_id_field
        self.clip_dimension = clip_dimension
        self.resnet_dimension = resnet_dimension
        self.text_vector_field = text_vector_field
        
        # Lấy thông tin kết nối từ biến môi trường nếu không được cung cấp
        self.uri = uri or get_env("MILVUS_URI", "http://localhost:19530")
        self.db_name = db_name or get_env("MILVUS_DB_NAME", "default")
        self.user = user or get_env("MILVUS_USER", "")
        self.password = password or get_env("MILVUS_PASSWORD", "")
        
        # Định nghĩa các trường metadata
        self.metadata_fields = metadata_fields or [
            ("metadata", DataType.JSON)  # Mặc định có một trường metadata dạng JSON
        ]
        
        try:
            # Kết nối tới Milvus server
            self._connect()
            
            # Kiểm tra và tạo collection nếu cần
            self._init_collection(recreate_collection)
        except MilvusException as e:
            print(f"Lỗi kết nối Milvus: {e}")
            print(f"Kiểm tra lại cấu hình kết nối:")
            print(f"- URI: {self.uri}")
            print(f"- Database: {self.db_name}")
            print(f"- User: {self.user}")
            raise
    
    def _connect(self) -> None:
        """
        Kết nối tới Milvus server
        """
        connections.connect(
            alias="default", 
            uri=self.uri,
            db_name=self.db_name,
            user=self.user,
            password=self.password
        )
    
    def _get_vector_dim(self) -> int:
        """
        Lấy số chiều của vector embedding
        
        Returns:
            int: Số chiều của vector hoặc -1 nếu không thể xác định.
        """
        if self.embedding_function:
            try:
                # Tạo một vector từ văn bản mẫu để xác định số chiều
                sample_vector = self.embedding_function("Sample text")
                if isinstance(sample_vector, list):
                    return len(sample_vector)
                elif hasattr(sample_vector, 'shape'):
                     return sample_vector.shape[0]
            except Exception as e:
                print(f"Warning: Không thể xác định số chiều từ embedding_function: {e}")
        # Trả về giá trị mặc định hoặc báo lỗi nếu không có embedding_function
        # Hoặc dựa vào collection schema nếu đã tồn tại? Tạm thời trả về -1
        print("Warning: Không thể xác định số chiều vector từ embedding_function.")
        return -1 # Hoặc một giá trị mặc định khác / raise error
    
    
    def _init_inventory_item_collection(self) -> None:
        """
        Khởi tạo collection inventory_item
        """
        # Xác định số chiều của vector
        dim = self._get_vector_dim()
        
        # Định nghĩa schema cho collection
        fields = [
            FieldSchema(name=self.id_field, dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=self.text_vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="inventory_item_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name=self.database_id_field, dtype=DataType.VARCHAR, max_length=36, is_partition_key=True)
        ]
        
        # Thêm các trường metadata
        for field_name, field_type in self.metadata_fields:
            fields.append(FieldSchema(name=field_name, dtype=field_type))
        
        # Tạo schema và collection
        try:
            schema = CollectionSchema(fields, enable_dynamic_field=False) # Không bật dynamic field trừ khi thực sự cần
            self.collection = Collection(self.collection_name, schema)
            print(f"Đã tạo collection '{self.collection_name}' với schema.")
        except Exception as e:
            print(f"Lỗi khi tạo collection '{self.collection_name}': {e}")
            raise # Ném lại lỗi để dừng quá trình nếu không tạo được collection

        # Tạo index cho các trường vector có trong schema
        index_params = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}}

        # Tạo index cho trường vector
        try:
            self.collection.create_index("vector", index_params)
            print("Đã tạo index cho trường 'vector'.")
        except Exception as e:
            print(f"Lỗi khi tạo index cho 'vector': {e}")

        # Load collection vào memory sau khi tạo index
        print(f"Đang load collection '{self.collection_name}'...")
        self.collection.load()
        print(f"Collection '{self.collection_name}' đã được load.")
    
    def _init_health_check_collection(self) -> None:
        """
        Khởi tạo collection health_check
        """
        self.collection = Collection(self.collection_name)
        self.collection.load()
    
    def _init_tool_collection(self) -> None:
        """
        Khởi tạo collection tool
        """
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
    def _init_collection(self, recreate: bool = False) -> None:
        """
        Khởi tạo collection trong Milvus
        
        Args:
            recreate: Có tạo lại collection nếu đã tồn tại không
        """
        # Kiểm tra collection đã tồn tại chưa
        if utility.has_collection(self.collection_name):
            if recreate:
                utility.drop_collection(self.collection_name)
            else:
                self.collection = Collection(self.collection_name)
                self.collection.load()
                self.collection = Collection(self.collection_name)
                print(f"Collection '{self.collection_name}' đã tồn tại.")
                return
            
        if self.collection_name == "inventory_item":
            self._init_inventory_item_collection()
            return

        if self.collection_name == "tool":
            self._init_inventory_item_collection()
            return
        
        if self.collection_name == "health_check":
            self._init_inventory_item_collection()
            return

        print(f"Tạo collection mới: {self.collection_name}")
        # Xác định số chiều của vector TEXT (nếu có embedding function)
        text_dim = self._get_vector_dim()
        if text_dim <= 0 and self.embedding_function:
             # Cố gắng lấy dimension từ cấu hình embedding nếu có
             if hasattr(self.embedding_function, 'client') and hasattr(self.embedding_function.client, 'dimensions'):
                 text_dim = self.embedding_function.client.dimensions
             else:
                 # Nếu vẫn không được, đặt giá trị mặc định hoặc báo lỗi nghiêm trọng hơn
                 print(f"Error: Không thể xác định số chiều cho vector text '{self.text_vector_field}'.")
                 # Có thể raise lỗi ở đây nếu trường text vector là bắt buộc
                 text_dim = 768 # Hoặc một giá trị mặc định an toàn khác

        # --- Định nghĩa schema dựa trên tên collection ---
        fields = []
        is_image_collection = (self.collection_name == "image_collection")

        # Trường ID luôn có
        fields.append(FieldSchema(name=self.id_field, dtype=DataType.VARCHAR, is_primary=True, max_length=300))

        # Thêm trường text và vector cho tất cả collection
        fields.append(FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=65535))
        fields.append(FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=text_dim))

        # Thêm các trường cơ bản
        fields.append(FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=65535))
        fields.append(FieldSchema(name="metadata", dtype=DataType.JSON, is_nullable=True))  # Thêm trường metadata và cho phép null
        fields.append(FieldSchema(name="database_id", dtype=DataType.VARCHAR, max_length=36))  # Thêm trường database_id

        # Thêm các trường metadata được định nghĩa trong metadata_fields (thường là 'metadata' JSON)
        for field_name, field_type in self.metadata_fields:
            # Đảm bảo không trùng tên với các trường đã định nghĩa ở trên
            defined_field_names = [f.name for f in fields]
            if field_name not in defined_field_names:
                 # Cần kiểm tra field_type là DataType hợp lệ
                 # Chuyển đổi string thành DataType nếu cần (ví dụ: "JSON" -> DataType.JSON)
                 actual_field_type = field_type
                 if isinstance(field_type, str):
                      try:
                           actual_field_type = getattr(DataType, field_type.upper())
                      except AttributeError:
                           print(f"Warning: Kiểu dữ liệu metadata không hợp lệ '{field_type}' cho trường '{field_name}'. Bỏ qua.")
                           continue # Bỏ qua field này

                 if isinstance(actual_field_type, DataType):
                     # Xử lý VARCHAR cần max_length
                     if actual_field_type == DataType.VARCHAR:
                          # Có thể cho phép tùy chỉnh max_length qua kwargs nếu cần
                          fields.append(FieldSchema(name=field_name, dtype=actual_field_type, max_length=65535))
                     else:
                          fields.append(FieldSchema(name=field_name, dtype=actual_field_type))
                 else:
                      print(f"Warning: Kiểu dữ liệu metadata không hợp lệ '{actual_field_type}' cho trường '{field_name}'. Bỏ qua.")


        # Tạo schema và collection
        try:
            schema = CollectionSchema(fields, enable_dynamic_field=False) # Không bật dynamic field trừ khi thực sự cần
            self.collection = Collection(self.collection_name, schema)
            print(f"Đã tạo collection '{self.collection_name}' với schema.")
        except Exception as e:
            print(f"Lỗi khi tạo collection '{self.collection_name}': {e}")
            raise # Ném lại lỗi để dừng quá trình nếu không tạo được collection

        # Tạo index cho các trường vector có trong schema
        index_params = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}}

        # Tạo index cho trường vector
        try:
            self.collection.create_index("vector", index_params)
            print("Đã tạo index cho trường 'vector'.")
        except Exception as e:
            print(f"Lỗi khi tạo index cho 'vector': {e}")

        # Load collection vào memory sau khi tạo index
        print(f"Đang load collection '{self.collection_name}'...")
        self.collection.load()
        print(f"Collection '{self.collection_name}' đã được load.")
        
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
        # Tạo ID nếu không được cung cấp
        ids = kwargs.get("ids", [str(i) for i in range(len(texts))])
        
        # Tạo metadata nếu không được cung cấp
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Tạo database_ids nếu không được cung cấp
        database_ids = kwargs.get("database_ids", [str(uuid.uuid4()) for _ in range(len(texts))])
        
        # Tạo vectors từ texts
        vectors = [self.embedding_function(text) for text in texts]
        
        # Chuẩn bị dữ liệu để chèn
        data = [
            ids,                   # ID field
            texts,                 # Text field
            vectors,               # Vector field
            database_ids,          # DatabaseId field
            metadatas,             # Metadata field
        ]
        
        # Chèn dữ liệu vào collection
        insert_result = self.collection.insert(data)
        self.collection.flush()
        
        return ids
    
    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """
        Thêm tài liệu vào vector store
        
        Args:
            documents: Danh sách tài liệu cần thêm
            
        Returns:
            List[str]: Danh sách ID của các tài liệu đã thêm
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        database_id: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm văn bản tương tự với query dựa trên độ tương đồng về vector
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            database_id: ID cơ sở dữ liệu để lọc kết quả (tùy chọn)
            
        Returns:
            List[Tuple[Any, float]]: Danh sách tuple gồm tài liệu và độ tương đồng
        """
        # Tạo vector từ query
        query_vector = self.embedding_function(query)
        
        # Thực hiện tìm kiếm
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        
        # Thêm bộ lọc nếu có database_id
        expr = None
        if database_id:
            expr = f'{self.database_id_field} == "{database_id}"'
        
        results = self.collection.search(
            data=[query_vector],
            anns_field=self.text_vector_field,
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=[self.text_field, self.database_id_field, "metadata"]
        )
        
        # Chuyển đổi kết quả sang định dạng trả về
        docs_with_scores = []
        for hit in results[0]:
            doc = {
                "id": hit.id,
                "text": hit.entity.get(self.text_field),
                "database_id": hit.entity.get(self.database_id_field),
                "metadata": hit.entity.get("metadata")
            }
            docs_with_scores.append((doc, hit.score))
        
        return docs_with_scores
    
    def fulltext_search(
        self, 
        query: str, 
        k: int = 4, 
        database_id: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm văn bản dựa trên full-text search
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            database_id: ID cơ sở dữ liệu để lọc kết quả (tùy chọn)
            
        Returns:
            List[Tuple[Any, float]]: Danh sách tuple gồm tài liệu và điểm số
        """
        try:
            # Thực hiện full-text search
            expr = f'{self.text_field} like "%{query}%"'
            
            # Thêm bộ lọc nếu có database_id
            if database_id:
                expr = f'({expr}) && ({self.database_id_field} == "{database_id}")'
            
            results = self.collection.query(
                expr=expr,
                output_fields=[self.id_field, self.text_field, self.database_id_field, "metadata"],
                limit=k
            )
            
            # Tính toán điểm số đơn giản dựa trên số lần xuất hiện của từ khóa
            docs_with_scores = []
            query_words = query.lower().split()
            
            for item in results:
                text = item.get(self.text_field, "").lower()
                score = 0
                for word in query_words:
                    score += text.count(word)
                
                # Chuẩn hóa điểm số
                if len(query_words) > 0:
                    score = score / len(query_words)
                
                doc = {
                    "id": item.get(self.id_field),
                    "text": item.get(self.text_field),
                    "database_id": item.get(self.database_id_field),
                    "metadata": item.get("metadata", {})
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
        database_id: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[Any, float]]:
        """
        Tìm kiếm văn bản kết hợp giữa vector similarity và full-text search
        
        Args:
            query: Câu truy vấn
            k: Số lượng kết quả trả về
            alpha: Trọng số cho kết quả vector similarity (0-1)
            database_id: ID cơ sở dữ liệu để lọc kết quả (tùy chọn)
            
        Returns:
            List[Tuple[Any, float]]: Danh sách tuple gồm tài liệu và điểm số kết hợp
        """
        # Lấy kết quả từ cả hai phương pháp
        vector_results = self.similarity_search(query, k=k*2, database_id=database_id, **kwargs)
        fulltext_results = self.fulltext_search(query, k=k*2, database_id=database_id, **kwargs)
        
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
    
    def delete(self, ids: List[str], **kwargs) -> None:
        """
        Xóa các văn bản khỏi vector store dựa trên ID
        
        Args:
            ids: Danh sách ID cần xóa
        """
        expr = f"{self.id_field} in {ids}"
        self.collection.delete(expr)
        self.collection.flush()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Lấy thông tin thống kê về collection
        
        Returns:
            Dict[str, Any]: Thông tin thống kê
        """
        stats = {
            "name": self.collection_name,
            "count": self.collection.num_entities,
            "fields": self.collection.schema.fields,
        }
        
        return stats
