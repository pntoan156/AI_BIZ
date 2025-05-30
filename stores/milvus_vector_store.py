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
        self.clip_dimension = clip_dimension
        self.resnet_dimension = resnet_dimension
        self.text_vector_field = text_vector_field
        
        # Lấy thông tin kết nối từ biến môi trường nếu không được cung cấp
        self.uri = uri or get_env("MILVUS_URI", "http://localhost:19530")
        self.db_name = db_name or get_env("MILVUS_DB_NAME", "default")
        print(f"MILVUS_DB_NAME: {self.db_name}")
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
        try:
            # Kết nối tới Milvus server với database đã chọn
            connections.connect(
                alias="default", 
                uri=self.uri,
                db_name=self.db_name,
                user=self.user,
                password=self.password
            )
            print(f"Đã kết nối tới database '{self.db_name}'")
        except Exception as e:
            # Nếu lỗi là do database không tồn tại, thử tạo mới
            if "database not found" in str(e).lower():
                # Kết nối lại không chỉ định database
                connections.connect(
                    alias="default", 
                    uri=self.uri,
                    user=self.user,
                    password=self.password
                )
                # Tạo database mới
                from pymilvus import db
                db.create_database(self.db_name)
                print(f"Đã tạo database '{self.db_name}'")
                
                # Kết nối lại với database mới
                connections.disconnect("default")
                connections.connect(
                    alias="default", 
                    uri=self.uri,
                    db_name=self.db_name,
                    user=self.user,
                    password=self.password
                )
                print(f"Đã kết nối tới database '{self.db_name}'")
            else:
                print(f"Lỗi khi kết nối: {e}")
                raise
    
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
    
    def _create_optimized_index(self, field_name: str = "vector") -> None:
        """
        Tạo index tối ưu cho vector field - SINGLE SOURCE OF TRUTH
        
        Args:
            field_name: Tên field cần đánh index (mặc định "vector")
        """
        # Tham số index tối ưu duy nhất - KHÔNG duplicate nữa
        index_params = {
            "metric_type": "COSINE", 
            "index_type": "HNSW", 
            "params": {
                "M": 16,             # Tối ưu cho chất lượng và hiệu suất
                "efConstruction": 200 # Tối ưu để giảm phân mảnh
            }
        }
        
        try:
            self.collection.create_index(field_name, index_params)
            print(f"✅ Đã tạo index tối ưu cho field '{field_name}' (M=16, efConstruction=200)")
        except Exception as e:
            print(f"❌ Lỗi khi tạo index cho '{field_name}': {e}")
            raise
    
    def _batch_embed_texts(self, texts: List[str], embedding_batch_size: int = 200) -> List[List[float]]:
        """
        Batch embedding với sub-batches để tránh quá tải API
        
        Args:
            texts: Danh sách texts cần embed
            embedding_batch_size: Kích thước sub-batch (mặc định 200)
            
        Returns:
            List[List[float]]: Danh sách vectors
        """
        import time  # Import time module
        
        if not texts:
            return []
            
        if len(texts) == 1:
            # Single text
            return [self.embedding_function(texts[0])]
        
        # Batch embedding với sub-batches
        all_vectors = []
        total_sub_batches = (len(texts) + embedding_batch_size - 1) // embedding_batch_size
        
        print(f"      🧠 Chia thành {total_sub_batches} sub-batches ({embedding_batch_size} records/batch)...")
        
        for sub_i in range(0, len(texts), embedding_batch_size):
            sub_batch_texts = texts[sub_i:sub_i + embedding_batch_size]
            sub_batch_num = (sub_i // embedding_batch_size) + 1
            
            sub_start = time.time()
            print(f"         🔄 Sub-batch {sub_batch_num}/{total_sub_batches}: {len(sub_batch_texts)} texts...")
            
            # Gọi embedding cho sub-batch
            sub_vectors = self.embedding_function(sub_batch_texts)
            all_vectors.extend(sub_vectors)
            
            sub_time = time.time() - sub_start
            sub_progress = (sub_batch_num / total_sub_batches) * 100
            sub_speed = len(sub_batch_texts) / sub_time if sub_time > 0 else 0
            
            print(f"         ✅ Sub-batch {sub_batch_num} hoàn thành ({sub_progress:.0f}%) - {sub_time:.1f}s - {sub_speed:.0f} texts/s")
        
        return all_vectors
    
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
            FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=100),  # Thêm trường image_id riêng
            FieldSchema(name="inventory_item_id", dtype=DataType.VARCHAR, max_length=100),
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

        # SỬ DỤNG METHOD DUY NHẤT để tạo index
        self._create_optimized_index("vector")

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

        # --- Định nghĩa schema tối ưu ---
        fields = []

        # Trường ID với kích thước hợp lý
        fields.append(FieldSchema(name=self.id_field, dtype=DataType.VARCHAR, is_primary=True, max_length=36))

        # Trường text với kích thước hợp lý hơn
        fields.append(FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=1000))
        
        # Vector field
        fields.append(FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=text_dim))

        # Thêm trường image_id riêng để lưu ID gốc của image
        fields.append(FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=100))

        # Chỉ thêm metadata JSON duy nhất - loại bỏ các trường dư thừa
        fields.append(FieldSchema(name="metadata", dtype=DataType.JSON, is_nullable=True))

        # Chỉ thêm các trường metadata từ metadata_fields nếu không trùng với 'metadata'
        for field_name, field_type in self.metadata_fields:
            if field_name != "metadata":  # Tránh trùng lặp
                defined_field_names = [f.name for f in fields]
                if field_name not in defined_field_names:
                     actual_field_type = field_type
                     if isinstance(field_type, str):
                          try:
                               actual_field_type = getattr(DataType, field_type.upper())
                          except AttributeError:
                               print(f"Warning: Kiểu dữ liệu metadata không hợp lệ '{field_type}' cho trường '{field_name}'. Bỏ qua.")
                               continue

                     if isinstance(actual_field_type, DataType):
                         if actual_field_type == DataType.VARCHAR:
                              fields.append(FieldSchema(name=field_name, dtype=actual_field_type, max_length=1024))  # Giảm max_length
                         else:
                              fields.append(FieldSchema(name=field_name, dtype=actual_field_type))

        # Tạo schema và collection
        try:
            schema = CollectionSchema(fields, enable_dynamic_field=False)
            self.collection = Collection(self.collection_name, schema)
            print(f"Đã tạo collection '{self.collection_name}' với schema tối ưu.")
        except Exception as e:
            print(f"Lỗi khi tạo collection '{self.collection_name}': {e}")
            raise

        # SỬ DỤNG METHOD DUY NHẤT để tạo index
        self._create_optimized_index("vector")

        # KHÔNG load collection ngay - sẽ load sau khi insert xong
        print(f"Collection '{self.collection_name}' đã được tạo. Sẽ load sau khi insert dữ liệu.")
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 5000,  # Tăng batch_size mặc định
        auto_flush: bool = True,  # Thêm tham số auto_flush
        **kwargs
    ) -> List[str]:
        """
        Thêm văn bản và metadata vào vector store với batch processing tối ưu
        
        Args:
            texts: Danh sách các văn bản cần thêm
            metadatas: Danh sách metadata tương ứng với mỗi văn bản
            batch_size: Kích thước batch để insert (mặc định 5000)
            auto_flush: Có tự động flush sau khi insert không (mặc định True)
            
        Returns:
            List[str]: Danh sách ID của các văn bản đã thêm
        """
        import time
        start_time = time.time()
        
        print(f"🚀 Bắt đầu add_texts: {len(texts):,} records với batch_size={batch_size:,}")
        
        # Tạo ID nếu không được cung cấp
        ids = kwargs.get("ids", [str(uuid.uuid4()) for _ in range(len(texts))])
        
        # Tạo metadata nếu không được cung cấp
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Đảm bảo collection đã được load
        collection_load_start = time.time()
        if not hasattr(self.collection, '_loaded') or not self.collection._loaded:
            try:
                self.collection.load()
                print(f"Đã load collection '{self.collection_name}' để insert dữ liệu.")
            except Exception as e:
                print(f"Warning: Không thể load collection: {e}")
        collection_load_time = time.time() - collection_load_start
        
        all_ids = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        embedding_time = 0
        insert_time = 0
        
        # Insert theo batch để tối ưu hiệu suất
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # Progress info
            progress = (batch_num / total_batches) * 100
            print(f"\n📦 BATCH {batch_num}/{total_batches} ({progress:.1f}%)")
            print(f"   📍 Processing: {i+1:,} → {min(i+len(batch_texts), len(texts)):,} của {len(texts):,}")
            
            # Đo thời gian embedding - WITH BATCH EMBEDDING
            embedding_start = time.time()
            print(f"   🧠 Batch embedding {len(batch_texts):,} texts...")
            
            # SỬ DỤNG BATCH EMBEDDING - NHANH HƠN NHIỀU LẦN!
            try:
                # Sử dụng helper method để batch embed
                batch_vectors = self._batch_embed_texts(batch_texts, embedding_batch_size=400)
                print(f"   ✅ BATCH EMBEDDING thành công ({len(batch_vectors)} vectors)!")
                    
            except Exception as batch_error:
                print(f"   ⚠️  Batch embedding failed: {batch_error}, fallback to single...")
                # Fallback về single embedding
                batch_vectors = []
                embedding_checkpoint = max(1, len(batch_texts) // 5)
                
                for j, text in enumerate(batch_texts):
                    vector = self.embedding_function(text)
                    batch_vectors.append(vector)
                    
                    if (j + 1) % embedding_checkpoint == 0 or j == len(batch_texts) - 1:
                        embed_progress = ((j + 1) / len(batch_texts)) * 100
                        embed_elapsed = time.time() - embedding_start
                        embed_speed = (j + 1) / embed_elapsed if embed_elapsed > 0 else 0
                        
                        print(f"      📊 {j+1:,}/{len(batch_texts):,} ({embed_progress:.0f}%) | "
                              f"Speed: {embed_speed:.0f}/s")
            
            batch_embedding_time = time.time() - embedding_start
            embedding_time += batch_embedding_time
            embedding_speed = len(batch_texts) / batch_embedding_time if batch_embedding_time > 0 else 0
            
            print(f"   ✅ Embedding: {batch_embedding_time:.2f}s ({embedding_speed:.0f} texts/s)")
            
            # Chuẩn bị dữ liệu để chèn (chỉ các trường cần thiết)
            data_prep_start = time.time()
            print(f"   🔧 Chuẩn bị insert data...")
            
            # Tạo image_ids từ metadata hoặc sử dụng batch_ids làm fallback
            batch_image_ids = []
            for i, metadata in enumerate(batch_metadatas):
                if isinstance(metadata, dict) and 'image_id' in metadata:
                    batch_image_ids.append(metadata['image_id'])
                elif isinstance(metadata, dict) and 'id' in metadata:
                    batch_image_ids.append(metadata['id'])  # Fallback cho legacy data
                else:
                    batch_image_ids.append(batch_ids[i])  # Fallback cuối cùng
            
            data = [
                batch_ids,           # ID field (primary key)
                batch_texts,         # Text field  
                batch_vectors,       # Vector field
                batch_image_ids,     # Image ID field (trường mới)
                batch_metadatas,     # Metadata field
            ]
            data_prep_time = time.time() - data_prep_start
            print(f"   ✅ Data prep: {data_prep_time:.3f}s")
            
            # Đo thời gian insert - WITH DETAILED LOGGING
            insert_start = time.time()
            print(f"   💾 Inserting {len(batch_texts):,} records...")
            try:
                insert_result = self.collection.insert(data)
                all_ids.extend(batch_ids)
                batch_insert_time = time.time() - insert_start
                insert_time += batch_insert_time
                
                # Insert performance details
                insert_speed = len(batch_texts) / batch_insert_time if batch_insert_time > 0 else 0
                data_size_mb = (len(batch_texts) * 1000) / (1024 * 1024)  # Rough estimate
                
                print(f"   ✅ Insert: {batch_insert_time:.2f}s ({insert_speed:.0f} rec/s)")
                print(f"      📊 ~{data_size_mb:.1f}MB | IDs: {len(insert_result.primary_keys):,}")
                
                # Batch summary
                batch_total_time = batch_embedding_time + batch_insert_time + data_prep_time
                records_per_sec = len(batch_texts) / batch_total_time if batch_total_time > 0 else 0
                
                print(f"   🎯 BATCH TOTAL: {batch_total_time:.2f}s | Speed: {records_per_sec:.0f} rec/s")
                print(f"      📈 Progress: {len(all_ids):,}/{len(texts):,} completed")
                
                # ETA calculation
                if batch_num > 1:
                    avg_batch_time = (embedding_time + insert_time) / batch_num
                    remaining_batches = total_batches - batch_num
                    eta_seconds = remaining_batches * avg_batch_time
                    eta_minutes = eta_seconds / 60
                    
                    print(f"      🕐 ETA: {eta_minutes:.1f} phút")
                    
                    # Mini progress bar
                    bar_length = 20
                    filled = int(bar_length * progress / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"      📊 [{bar}] {progress:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Lỗi insert batch {batch_num}: {e}")
                raise
        
        # Đo thời gian flush
        flush_start = time.time()
        if auto_flush:
            try:
                self.collection.flush()
                flush_time = time.time() - flush_start
                print(f"✅ Đã flush {len(all_ids):,} records trong {flush_time:.2f}s")
            except Exception as e:
                print(f"Warning: Lỗi khi flush: {e}")
        else:
            flush_time = 0
        
        # Tính tổng thời gian và thống kê
        total_time = time.time() - start_time
        overall_speed = len(all_ids) / total_time
        
        print(f"🎉 Hoàn thành add_texts!")
        print(f"📊 THỐNG KÊ HIỆU SUẤT:")
        print(f"   📝 Records: {len(all_ids):,}/{len(texts):,}")
        print(f"   ⏰ Tổng thời gian: {total_time:.2f}s")
        print(f"   🔄 Collection load: {collection_load_time:.2f}s")
        print(f"   🧠 Embedding: {embedding_time:.2f}s ({embedding_time/total_time*100:.1f}%)")
        print(f"   💾 Insert: {insert_time:.2f}s ({insert_time/total_time*100:.1f}%)")
        print(f"   🚀 Flush: {flush_time:.2f}s ({flush_time/total_time*100:.1f}%)")
        print(f"   ⚡ Tốc độ: {overall_speed:.0f} records/second")
        
        return all_ids
    
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
        # Tạo vector từ query
        query_vector = self.embedding_function(query)
        
        # Thực hiện tìm kiếm
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        
        expr = None
        
        results = self.collection.search(
            data=[query_vector],
            anns_field=self.text_vector_field,
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=[self.text_field, "metadata"]
        )
        
        # Chuyển đổi kết quả sang định dạng trả về
        docs_with_scores = []
        for hit in results[0]:
            doc = {
                "id": hit.id,
                "text": hit.entity.get(self.text_field),
                "metadata": hit.entity.get("metadata")
            }
            docs_with_scores.append((doc, hit.score))
        
        return docs_with_scores
    
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
            expr = f'{self.text_field} like "%{query}%"'
            
            results = self.collection.query(
                expr=expr,
                output_fields=[self.id_field, self.text_field, "metadata"],
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

    def optimize_collection(self) -> None:
        """
        Tối ưu hóa collection sau khi insert dữ liệu xong
        - Compact để giảm phân mảnh
        - Rebuild index nếu cần
        """
        try:
            print(f"Đang tối ưu hóa collection '{self.collection_name}'...")
            
            # Flush để đảm bảo tất cả dữ liệu đã được ghi
            self.collection.flush()
            
            # Compact để giảm phân mảnh
            self.collection.compact()
            print("Đã thực hiện compact collection.")
            
            # Kiểm tra và rebuild index nếu cần
            index_info = self.collection.indexes
            if index_info:
                print("Index đã tồn tại, không cần rebuild.")
            else:
                print("Đang rebuild index...")
                # SỬ DỤNG METHOD DUY NHẤT để tạo index
                self._create_optimized_index("vector")
            
            # Load collection để sử dụng
            self.collection.load()
            print(f"Collection '{self.collection_name}' đã được tối ưu hóa và load.")
            
        except Exception as e:
            print(f"Lỗi khi tối ưu hóa collection: {e}")

    def bulk_insert_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 10000,  # Tăng batch size cho bulk insert
        **kwargs
    ) -> List[str]:
        """
        Insert hàng loạt với tối ưu hóa cao nhất
        
        Args:
            texts: Danh sách văn bản
            metadatas: Danh sách metadata
            batch_size: Kích thước batch (mặc định 10000)
            
        Returns:
            List[str]: Danh sách ID đã insert
        """
        import time
        start_time = time.time()
        
        print(f"🚀 Bắt đầu BULK INSERT: {len(texts):,} records với batch_size={batch_size:,}")
        
        # Insert tất cả batch mà không flush
        insert_start = time.time()
        all_ids = self.add_texts(
            texts=texts, 
            metadatas=metadatas, 
            batch_size=batch_size,
            auto_flush=False,  # Không flush từng batch
            **kwargs
        )
        insert_time = time.time() - insert_start
        
        # Flush một lần duy nhất ở cuối
        flush_start = time.time()
        print("🔄 Đang flush tất cả dữ liệu...")
        self.collection.flush()
        flush_time = time.time() - flush_start
        
        # Tối ưu hóa collection
        optimize_start = time.time()
        print("⚡ Đang tối ưu hóa collection...")
        self.optimize_collection()
        optimize_time = time.time() - optimize_start
        
        # Tính tổng thời gian
        total_time = time.time() - start_time
        overall_speed = len(all_ids) / total_time
        
        print(f"🎉 HOÀN THÀNH BULK INSERT!")
        print(f"📊 THỐNG KÊ TỔNG QUAN:")
        print(f"   📝 Records: {len(all_ids):,}")
        print(f"   ⏰ Tổng thời gian: {total_time:.2f}s")
        print(f"   💾 Insert + Embedding: {insert_time:.2f}s ({insert_time/total_time*100:.1f}%)")
        print(f"   🚀 Final Flush: {flush_time:.2f}s ({flush_time/total_time*100:.1f}%)")
        print(f"   ⚙️  Optimization: {optimize_time:.2f}s ({optimize_time/total_time*100:.1f}%)")
        print(f"   ⚡ Tốc độ tổng: {overall_speed:.0f} records/second")
        
        return all_ids

    def mega_insert_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 50000,  # Batch rất lớn cho mega insert
        **kwargs
    ) -> List[str]:
        """
        Insert siêu lớn cho hàng triệu bản ghi với tối ưu hóa tối đa
        
        Args:
            texts: Danh sách văn bản
            metadatas: Danh sách metadata  
            batch_size: Kích thước batch (mặc định 50,000)
            
        Returns:
            List[str]: Danh sách ID đã insert
        """
        import time
        start_time = time.time()
        
        total_records = len(texts)
        print(f"🚀 Bắt đầu MEGA INSERT {total_records:,} records với batch_size={batch_size:,}")
        
        if total_records > 1000000:  # > 1 triệu
            print("⚠️  CẢNH BÁO: Insert hơn 1 triệu records. Đảm bảo đủ RAM và thời gian!")
        
        # Tạo ID nếu không được cung cấp
        preparation_start = time.time()
        ids = kwargs.get("ids", [str(uuid.uuid4()) for _ in range(total_records)])
        
        # Tạo metadata nếu không được cung cấp
        if metadatas is None:
            metadatas = [{} for _ in texts]
        preparation_time = time.time() - preparation_start
        
        # Đảm bảo collection đã được load
        load_start = time.time()
        if not hasattr(self.collection, '_loaded') or not self.collection._loaded:
            try:
                self.collection.load()
                print(f"Đã load collection '{self.collection_name}' để insert dữ liệu.")
            except Exception as e:
                print(f"Warning: Không thể load collection: {e}")
        load_time = time.time() - load_start
        
        all_ids = []
        total_batches = (total_records + batch_size - 1) // batch_size
        total_embedding_time = 0
        total_insert_time = 0
        
        # Insert theo batch siêu lớn
        for i in range(0, total_records, batch_size):
            batch_num = i // batch_size + 1
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            batch_start = time.time()
            progress = (batch_num / total_batches) * 100
            print(f"\n📦 BATCH {batch_num}/{total_batches} ({progress:.1f}%) - {len(batch_texts):,} records")
            print(f"   📍 Records: {i+1:,} → {min(i+len(batch_texts), total_records):,}")
            
            # Tạo vectors từ batch texts - WITH BATCH EMBEDDING
            embedding_start = time.time()
            print(f"   🧠 Batch embedding {len(batch_texts):,} texts...")
            
            # SỬ DỤNG BATCH EMBEDDING - SIÊU NHANH!
            try:
                # Sử dụng helper method để batch embed
                batch_vectors = self._batch_embed_texts(batch_texts, embedding_batch_size=400)
                print(f"   🚀 BATCH EMBEDDING thành công ({len(batch_vectors)} vectors) - siêu tối ưu!")
                    
            except Exception as batch_error:
                print(f"   ⚠️  Batch embedding error: {batch_error}, fallback...")
                # Fallback to single embedding với progress tracking
                batch_vectors = []
                embedding_checkpoint = max(1, len(batch_texts) // 4)
                
                for j, text in enumerate(batch_texts):
                    vector = self.embedding_function(text)
                    batch_vectors.append(vector)
                    
                    if (j + 1) % embedding_checkpoint == 0 or j == len(batch_texts) - 1:
                        embed_progress = ((j + 1) / len(batch_texts)) * 100
                        embed_elapsed = time.time() - embedding_start
                        embed_speed = (j + 1) / embed_elapsed
                        remaining_embeds = len(batch_texts) - (j + 1)
                        embed_eta = remaining_embeds / embed_speed if embed_speed > 0 else 0
                        
                        print(f"      📊 Embedding: {j+1:,}/{len(batch_texts):,} ({embed_progress:.0f}%) | "
                              f"Speed: {embed_speed:.0f}/s | ETA: {embed_eta:.1f}s")
            
            embedding_time = time.time() - embedding_start
            total_embedding_time += embedding_time
            embedding_speed = len(batch_texts) / embedding_time
            
            print(f"   ✅ Embedding hoàn thành: {embedding_time:.1f}s ({embedding_speed:.0f} texts/s)")
            
            # Chuẩn bị dữ liệu để chèn
            data_prep_start = time.time()
            print(f"   🔧 Chuẩn bị dữ liệu để insert...")
            
            # Tạo image_ids từ metadata hoặc sử dụng batch_ids làm fallback
            batch_image_ids = []
            for j, metadata in enumerate(batch_metadatas):
                if isinstance(metadata, dict) and 'image_id' in metadata:
                    batch_image_ids.append(metadata['image_id'])
                elif isinstance(metadata, dict) and 'id' in metadata:
                    batch_image_ids.append(metadata['id'])  # Fallback cho legacy data
                else:
                    batch_image_ids.append(batch_ids[j])  # Fallback cuối cùng
            
            data = [
                batch_ids,           # ID field (primary key - UUID)
                batch_texts,         # Text field
                batch_vectors,       # Vector field
                batch_image_ids,     # Image ID field (trường mới)
                batch_metadatas,     # Metadata field
            ]
            data_prep_time = time.time() - data_prep_start
            print(f"   ✅ Dữ liệu đã chuẩn bị: {data_prep_time:.2f}s")
            
            # Chèn batch vào collection - WITH DETAILED LOGGING  
            insert_start = time.time()
            print(f"   💾 Bắt đầu insert {len(batch_texts):,} records vào Milvus...")
            
            try:
                insert_result = self.collection.insert(data)
                all_ids.extend(batch_ids)
                insert_time = time.time() - insert_start
                total_insert_time += insert_time
                
                # Chi tiết về insert performance
                insert_speed = len(batch_texts) / insert_time
                data_size_mb = (len(batch_texts) * (1000 + 512 * 4)) / (1024 * 1024)  # Ước tính MB
                
                print(f"   ✅ Insert hoàn thành: {insert_time:.1f}s ({insert_speed:.0f} rec/s)")
                print(f"      📊 Data size: ~{data_size_mb:.1f}MB | Insert IDs: {len(insert_result.primary_keys):,}")
                
                # Overall batch performance
                batch_total_time = time.time() - batch_start
                batch_speed = len(batch_texts) / batch_total_time
                
                print(f"   🎯 BATCH SUMMARY:")
                print(f"      ⏱️  Total: {batch_total_time:.1f}s | Embedding: {embedding_time:.1f}s ({embedding_time/batch_total_time*100:.0f}%) | Insert: {insert_time:.1f}s ({insert_time/batch_total_time*100:.0f}%)")
                print(f"      ⚡ Speed: {batch_speed:.0f} rec/s total | {len(all_ids):,}/{total_records:,} completed")
                
                # Ước tính thời gian còn lại - IMPROVED ETA
                if batch_num > 1:
                    avg_time_per_batch = (time.time() - start_time - preparation_time - load_time) / batch_num
                    remaining_batches = total_batches - batch_num
                    eta_seconds = remaining_batches * avg_time_per_batch
                    eta_minutes = eta_seconds / 60
                    eta_hours = eta_minutes / 60
                    
                    if eta_hours >= 1:
                        print(f"      🕐 ETA: {eta_hours:.1f} giờ ({eta_minutes:.0f} phút)")
                    else:
                        print(f"      🕐 ETA: {eta_minutes:.1f} phút")
                    
                    # Progress bar visual
                    bar_length = 30
                    filled_length = int(bar_length * progress / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"      📈 [{bar}] {progress:.1f}%")
                
            except Exception as e:
                print(f"   ❌ Lỗi khi insert batch {batch_num}: {e}")
                raise
        
        # Flush một lần duy nhất ở cuối
        flush_start = time.time()
        print("🔄 Đang flush tất cả dữ liệu...")
        self.collection.flush()
        flush_time = time.time() - flush_start
        
        # Tối ưu hóa collection
        optimize_start = time.time()
        print("⚡ Đang tối ưu hóa collection...")
        self.optimize_collection()
        optimize_time = time.time() - optimize_start
        
        # Tính toán thống kê tổng quan
        total_time = time.time() - start_time
        overall_speed = len(all_ids) / total_time
        
        print(f"🎉 HOÀN THÀNH MEGA INSERT {len(all_ids):,} records!")
        print(f"📊 THỐNG KÊ MEGA INSERT:")
        print(f"   📝 Records: {len(all_ids):,}")
        print(f"   ⏰ Tổng thời gian: {total_time/60:.1f} phút ({total_time:.1f}s)")
        print(f"   🔧 Preparation: {preparation_time:.1f}s ({preparation_time/total_time*100:.1f}%)")
        print(f"   🔄 Collection load: {load_time:.1f}s ({load_time/total_time*100:.1f}%)")
        print(f"   🧠 Total Embedding: {total_embedding_time:.1f}s ({total_embedding_time/total_time*100:.1f}%)")
        print(f"   💾 Total Insert: {total_insert_time:.1f}s ({total_insert_time/total_time*100:.1f}%)")
        print(f"   🚀 Final Flush: {flush_time:.1f}s ({flush_time/total_time*100:.1f}%)")
        print(f"   ⚙️  Optimization: {optimize_time:.1f}s ({optimize_time/total_time*100:.1f}%)")
        print(f"   ⚡ Tốc độ trung bình: {overall_speed:.0f} records/second")
        print(f"   💰 Chi phí thời gian trên 1K records: {total_time/total_records*1000:.2f}s")
        
        return all_ids

    def upsert_image(
        self,
        image_id: str,
        image_name: str,
        image_path: str,
        category: str,
        style: str = "",
        app_name: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Upsert (insert or update) một image record trong Milvus
        
        Args:
            image_id: ID của image (primary key)
            image_name: Tên của image (sẽ được embedding)
            image_path: Đường dẫn file image
            category: Danh mục
            style: Style của image
            app_name: Tên app
            
        Returns:
            Dict[str, Any]: Kết quả upsert với thông tin chi tiết
        """
        import time
        start_time = time.time()
        
        print(f"🔄 Upsert image: {image_id} - {image_name[:50]}...")
        
        try:
            # Đảm bảo collection đã được load
            if not hasattr(self.collection, '_loaded') or not self.collection._loaded:
                self.collection.load()
            
            # Kiểm tra xem record đã tồn tại chưa (theo image_id field)
            check_start = time.time()
            existing_expr = f"image_id == '{image_id}'"
            existing_results = self.collection.query(
                expr=existing_expr,
                output_fields=[self.id_field, self.text_field, "image_id", "metadata"],
                limit=1
            )
            check_time = time.time() - check_start
            
            is_update = len(existing_results) > 0
            action_type = "UPDATE" if is_update else "INSERT"
            
            print(f"   🔍 Check existence: {check_time:.3f}s - {action_type}")
            
            # Merge dữ liệu cũ với dữ liệu mới
            merge_start = time.time()
            if is_update:
                # Lấy dữ liệu cũ
                old_record = existing_results[0]
                old_metadata = old_record.get("metadata", {})
                old_text = old_record.get(self.text_field, "")
                old_primary_key = old_record.get(self.id_field, "")  # Giữ nguyên primary key cũ
                
                print(f"   📋 Merging with existing data...")
                print(f"      Primary Key: {old_primary_key} (preserved)")
                print(f"      Old: {old_text[:30]}...")
                print(f"      New: {image_name[:30]}...")
                
                # Merge metadata: Giữ dữ liệu cũ, override với dữ liệu mới
                merged_metadata = old_metadata.copy()
                merged_metadata.update({
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_name": image_name,
                    "category": category,
                    "style": style,
                    "app_name": app_name
                })
                
                # Sử dụng dữ liệu merged với primary key cũ
                final_primary_key = old_primary_key  # Giữ nguyên UUID cũ
                final_image_name = image_name
                final_metadata = merged_metadata
                
                # Xóa record cũ sau khi đã lấy dữ liệu
                delete_start = time.time()
                print(f"   🗑️  Deleting old record...")
                self.collection.delete(existing_expr)
                delete_time = time.time() - delete_start
                print(f"   ✅ Delete completed: {delete_time:.3f}s")
            else:
                # Dữ liệu hoàn toàn mới - tạo UUID mới
                import uuid
                final_primary_key = str(uuid.uuid4())  # UUID mới cho INSERT
                final_image_name = image_name
                final_metadata = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_name": image_name,
                    "category": category,
                    "style": style,
                    "app_name": app_name
                }
                delete_time = 0
                print(f"   🆕 New record with UUID: {final_primary_key}")
            
            merge_time = time.time() - merge_start
            print(f"   🔀 Data merge: {merge_time:.3f}s")
            
            # Tạo embedding cho final_image_name
            embedding_start = time.time()
            print(f"   🧠 Creating embedding for: {final_image_name}")
            image_vector = self.embedding_function(final_image_name)
            embedding_time = time.time() - embedding_start
            print(f"   ✅ Embedding: {embedding_time:.3f}s ({len(image_vector)} dims)")
            
            # Chuẩn bị dữ liệu để insert với primary key đúng
            data_prep_start = time.time()
            data = [
                [final_primary_key],  # ID field (UUID cũ nếu UPDATE, UUID mới nếu INSERT)
                [final_image_name],   # Text field (đã merge)
                [image_vector],       # Vector field
                [image_id],           # Image ID field (business ID)
                [final_metadata]      # Metadata field (đã merge)
            ]
            data_prep_time = time.time() - data_prep_start
            
            # Insert record mới
            insert_start = time.time()
            print(f"   💾 Inserting new record...")
            insert_result = self.collection.insert(data)
            insert_time = time.time() - insert_start
            print(f"   ✅ Insert: {insert_time:.3f}s")
            
            # Flush để đảm bảo dữ liệu được ghi
            flush_start = time.time()
            self.collection.flush()
            flush_time = time.time() - flush_start
            print(f"   🚀 Flush: {flush_time:.3f}s")
            
            # Tính toán tổng thời gian
            total_time = time.time() - start_time
            
            print(f"🎉 {action_type} completed!")
            print(f"   ⏰ Total time: {total_time:.3f}s")
            print(f"   📊 Check: {check_time:.3f}s | Merge: {merge_time:.3f}s | Embedding: {embedding_time:.3f}s | Insert: {insert_time:.3f}s | Flush: {flush_time:.3f}s")
            
            return {
                "success": True,
                "action": action_type.lower(),
                "primary_key": final_primary_key,  # UUID thực tế trong DB
                "image_id": image_id,              # Business ID
                "image_name": final_image_name,    # Sử dụng final name
                "is_update": is_update,
                "timing": {
                    "total_time": total_time,
                    "check_time": check_time,
                    "merge_time": merge_time,
                    "delete_time": delete_time if is_update else 0,
                    "embedding_time": embedding_time,
                    "insert_time": insert_time,
                    "flush_time": flush_time
                },
                "metadata": final_metadata,
                "insert_result": {
                    "primary_keys": len(insert_result.primary_keys),
                    "insert_count": insert_result.insert_count
                }
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"❌ Upsert failed for {image_id}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "action": "error",
                "image_id": image_id,
                "image_name": image_name,
                "error": str(e),
                "timing": {
                    "total_time": total_time
                }
            }

    def get_image_by_id(self, image_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin image theo image_id (không phải primary key)
        
        Args:
            image_id: ID của image trong trường image_id
            
        Returns:
            Dict[str, Any]: Thông tin image hoặc None nếu không tìm thấy
        """
        try:
            # Đảm bảo collection đã được load
            if not hasattr(self.collection, '_loaded') or not self.collection._loaded:
                self.collection.load()
            
            # Query để lấy image theo image_id field (không phải primary key)
            expr = f"image_id == '{image_id}'"
            results = self.collection.query(
                expr=expr,
                output_fields=[self.id_field, self.text_field, "image_id", "metadata"],
                limit=1
            )
            
            if results:
                result = results[0]
                metadata = result.get("metadata", {})
                
                return {
                    "success": True,
                    "found": True,
                    "data": {
                        "id": result.get(self.id_field),  # Primary key UUID
                        "image_id": result.get("image_id", ""),  # Trường image_id 
                        "text": result.get(self.text_field),
                        "image_path": metadata.get("image_path", ""),
                        "image_name": metadata.get("image_name", ""),
                        "category": metadata.get("category", ""),
                        "style": metadata.get("style", ""),
                        "app_name": metadata.get("app_name", ""),
                        "metadata": metadata
                    }
                }
            else:
                return {
                    "success": True,
                    "found": False,
                    "message": f"Image với image_id '{image_id}' không tồn tại"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_id": image_id
            }