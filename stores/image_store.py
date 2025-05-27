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

    def _check_existing_ids(self, ids: List[str]) -> Tuple[List[str], List[str]]:
        """
        Kiểm tra ID nào đã tồn tại trong collection
        
        Args:
            ids: Danh sách ID cần kiểm tra
            
        Returns:
            Tuple[List[str], List[str]]: (existing_ids, new_ids)
        """
        try:
            # Tạo expression để query các ID
            ids_str = "', '".join(ids)
            expr = f"id in ['{ids_str}']"
            
            results = self.vectorstore.collection.query(
                expr=expr,
                output_fields=["id"],
                limit=len(ids)
            )
            
            existing_ids = [item["id"] for item in results]
            new_ids = [id for id in ids if id not in existing_ids]
            
            return existing_ids, new_ids
            
        except Exception as e:
            print(f"Lỗi khi kiểm tra existing IDs: {e}")
            # Nếu có lỗi, coi như tất cả đều là ID mới
            return [], ids

    def _batch_embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embedding texts theo batch để tối ưu hiệu suất
        
        Args:
            texts: Danh sách văn bản cần embedding
            batch_size: Kích thước mỗi batch
            
        Returns:
            List[List[float]]: Danh sách vectors
        """
        all_vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Đang embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                # Sử dụng embed_documents để embedding cả batch
                batch_vectors = self.embedding_function(batch_texts)
                all_vectors.extend(batch_vectors)
                
                # Nghỉ ngắn giữa các batch để tránh quá tải API
                if i + batch_size < len(texts):
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Lỗi khi embedding batch {i//batch_size + 1}: {e}")
                # Fallback về embedding từng text
                for text in batch_texts:
                    try:
                        vector = self.embedding_function(text)
                        all_vectors.append(vector)
                    except Exception as text_error:
                        print(f"Lỗi khi embedding text: {text_error}")
                        # Tạo vector zero nếu không thể embedding
                        all_vectors.append([0.0] * 768)
        
        return all_vectors

    def _prepare_bulk_data(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]], 
        vectors: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Chuẩn bị dữ liệu cho bulk insert
        
        Args:
            texts: Danh sách văn bản
            metadatas: Danh sách metadata
            vectors: Danh sách vectors
            
        Returns:
            Dict: Dữ liệu bulk insert format
        """
        bulk_data = {
            "rows": []
        }
        
        for text, metadata, vector in zip(texts, metadatas, vectors):
            row = {
                "id": metadata.get("id", ""),
                "text": text,
                "vector": vector,
                "image_path": metadata.get("image_path", ""),
                "category": metadata.get("category", ""),
                "style": metadata.get("style", ""),
                "app_name": metadata.get("app_name", ""),
                "metadata": metadata
            }
            bulk_data["rows"].append(row)
        
        return bulk_data

    def _prepare_bulk_file_path(self, local_file_path: str, chunk_index: int) -> str:
        """
        Chuẩn bị đường dẫn file cho bulk insert
        
        Args:
            local_file_path: Đường dẫn file local
            chunk_index: Index của chunk
            
        Returns:
            str: Đường dẫn file để sử dụng cho bulk insert
        """
        try:
            # Kiểm tra có sử dụng MinIO không
            use_minio = get_env("USE_MINIO", "false").lower() == "true"
            
            if use_minio:
                return self._upload_file_to_minio(local_file_path, chunk_index)
            else:
                # Sử dụng local file path trực tiếp
                # Tạo thư mục bulk_data nếu chưa có
                bulk_dir = os.path.join(os.getcwd(), "bulk_data")
                os.makedirs(bulk_dir, exist_ok=True)
                
                # Copy file vào thư mục bulk_data với tên chuẩn
                import shutil
                target_file = os.path.join(bulk_dir, f"chunk_{chunk_index}_{int(time.time())}.json")
                shutil.copy2(local_file_path, target_file)
                
                print(f"Đã chuẩn bị file local: {target_file}")
                return target_file
                
        except Exception as e:
            print(f"Lỗi khi chuẩn bị file path: {e}")
            return local_file_path

    def _upload_file_to_minio(self, local_file_path: str, chunk_index: int) -> str:
        """
        Upload file lên MinIO server (tùy chọn)
        
        Args:
            local_file_path: Đường dẫn file local
            chunk_index: Index của chunk
            
        Returns:
            str: Đường dẫn file trên MinIO hoặc local nếu thất bại
        """
        try:
            from minio import Minio
            from minio.error import S3Error
            
            # Lấy thông tin MinIO từ env
            minio_endpoint = get_env("MINIO_ENDPOINT", "localhost:9000")
            minio_access_key = get_env("MINIO_ACCESS_KEY", "minioadmin")
            minio_secret_key = get_env("MINIO_SECRET_KEY", "minioadmin")
            minio_bucket = get_env("MINIO_BUCKET", "a-bucket")
            
            # Tạo MinIO client
            client = Minio(
                minio_endpoint,
                access_key=minio_access_key,
                secret_key=minio_secret_key,
                secure=False  # Set True nếu sử dụng HTTPS
            )
            
            # Kiểm tra bucket có tồn tại không
            if not client.bucket_exists(minio_bucket):
                client.make_bucket(minio_bucket)
                print(f"Đã tạo bucket: {minio_bucket}")
            
            # Tạo remote file path
            remote_file_path = f"bulk_insert/chunk_{chunk_index}_{int(time.time())}.json"
            
            # Upload file
            client.fput_object(
                minio_bucket,
                remote_file_path,
                local_file_path,
                content_type="application/json"
            )
            
            print(f"Đã upload file lên MinIO: {remote_file_path}")
            return remote_file_path
            
        except Exception as e:
            print(f"Lỗi khi upload file lên MinIO: {e}")
            print("Fallback về sử dụng local file")
            return local_file_path

    def bulk_insert_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 50000,
        **kwargs
    ) -> List[str]:
        """
        Bulk insert văn bản vào store - tối ưu cho dữ liệu lớn
        
        Args:
            texts: Danh sách văn bản
            metadatas: Danh sách metadata tương ứng
            chunk_size: Kích thước chunk cho bulk insert
            **kwargs: Các tham số bổ sung
            
        Returns:
            List[str]: Danh sách ID của các văn bản đã thêm
        """
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        try:
            # Import utility từ pymilvus
            from pymilvus import utility
            
            # Tạo ID cho các bản ghi
            ids = [metadata.get("id", str(i)) for i, metadata in enumerate(metadatas)]
            
            # Kiểm tra ID nào đã tồn tại
            existing_ids, new_ids = self._check_existing_ids(ids)
            print(f"Tìm thấy {len(existing_ids)} ID đã tồn tại, {len(new_ids)} ID mới")
            
            # Xóa các bản ghi cũ nếu có
            if existing_ids:
                print(f"Đang xóa {len(existing_ids)} bản ghi cũ...")
                self.delete(existing_ids)
                time.sleep(1)  # Đợi xóa hoàn tất
            
            # Embedding tất cả texts theo batch
            print("Bắt đầu embedding texts...")
            vectors = self._batch_embed_texts(texts, batch_size=100)
            
            # Xử lý bulk insert theo chunk
            all_inserted_ids = []
            total_chunks = (len(texts) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i:i + chunk_size]
                chunk_metadatas = metadatas[i:i + chunk_size]
                chunk_vectors = vectors[i:i + chunk_size]
                chunk_ids = ids[i:i + chunk_size]
                
                print(f"Đang bulk insert chunk {i//chunk_size + 1}/{total_chunks} ({len(chunk_texts)} items)")
                
                # Chuẩn bị dữ liệu bulk
                bulk_data = self._prepare_bulk_data(chunk_texts, chunk_metadatas, chunk_vectors)
                
                # Tạo file tạm cho bulk insert
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    json.dump(bulk_data, temp_file, ensure_ascii=False, indent=2)
                    temp_file_path = temp_file.name
                
                try:
                    # Chuẩn bị file path cho bulk insert
                    bulk_file_path = self._prepare_bulk_file_path(temp_file_path, i//chunk_size + 1)
                    
                    # Thực hiện bulk insert sử dụng utility.do_bulk_insert
                    print(f"Bulk inserting từ file: {bulk_file_path}")
                    
                    task_id = utility.do_bulk_insert(
                        collection_name=self.collection_name,
                        files=[bulk_file_path]
                    )
                    
                    print(f"Bulk insert task ID: {task_id}")
                    
                    # Đợi task hoàn thành
                    self._wait_for_bulk_insert_completion(task_id)
                    
                    all_inserted_ids.extend(chunk_ids)
                    print(f"Đã bulk insert thành công chunk {i//chunk_size + 1}")
                    
                except Exception as bulk_error:
                    print(f"Lỗi bulk insert chunk {i//chunk_size + 1}: {bulk_error}")
                    print("Fallback về insert thường...")
                    
                    # Fallback về insert thường
                    chunk_data = []
                    for j, (text, metadata, vector, image_id) in enumerate(zip(chunk_texts, chunk_metadatas, chunk_vectors, chunk_ids)):
                        record = {
                            "id": image_id,
                            "vector": vector,
                            "text": text,
                            "image_path": metadata.get("image_path", ""),
                            "category": metadata.get("category", ""),
                            "style": metadata.get("style", ""),
                            "app_name": metadata.get("app_name", ""),
                            "metadata": metadata
                        }
                        chunk_data.append(record)
                    
                    # Insert thường
                    insert_result = self.vectorstore.collection.insert(chunk_data)
                    all_inserted_ids.extend(chunk_ids)
                    self.vectorstore.collection.flush()
                    
                finally:
                    # Xóa file tạm
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                
                # Nghỉ giữa các chunk
                if i + chunk_size < len(texts):
                    print("Nghỉ 3 giây trước chunk tiếp theo...")
                    time.sleep(3)
            
            print(f"Hoàn thành bulk insert! Đã xử lý {len(all_inserted_ids)}/{len(texts)} items")
            return all_inserted_ids
            
        except Exception as e:
            print(f"Lỗi khi bulk insert texts: {str(e)}")
            print("Fallback về insert thường...")
            # Fallback về phương thức insert thường
            return self.add_texts(texts, metadatas, batch_size=1000)

    def _wait_for_bulk_insert_completion(self, task_id: str, timeout: int = 300):
        """
        Đợi bulk insert task hoàn thành
        
        Args:
            task_id: ID của bulk insert task
            timeout: Timeout tính bằng giây
        """
        from pymilvus import utility
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Kiểm tra trạng thái task
                task_state = utility.get_bulk_insert_state(task_id)
                
                if task_state.state_name == "Completed":
                    print(f"Bulk insert task {task_id} hoàn thành thành công")
                    return
                elif task_state.state_name == "Failed":
                    raise Exception(f"Bulk insert task {task_id} thất bại: {task_state.failed_reason}")
                else:
                    print(f"Bulk insert task {task_id} đang xử lý... ({task_state.state_name})")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Lỗi khi kiểm tra trạng thái bulk insert: {e}")
                time.sleep(5)
        
        raise Exception(f"Bulk insert task {task_id} timeout sau {timeout} giây")

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 50,
        **kwargs
    ) -> List[str]:
        """
        Thêm văn bản vào store với batch processing và update mechanism
        
        Args:
            texts: Danh sách văn bản
            metadatas: Danh sách metadata tương ứng
            batch_size: Kích thước mỗi batch để insert/update
            **kwargs: Các tham số bổ sung
            
        Returns:
            List[str]: Danh sách ID của các văn bản đã thêm/cập nhật
        """
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        try:
            # Tạo ID cho các bản ghi
            ids = [metadata.get("id", str(i)) for i, metadata in enumerate(metadatas)]
            
            # Kiểm tra ID nào đã tồn tại
            existing_ids, new_ids = self._check_existing_ids(ids)
            print(f"Tìm thấy {len(existing_ids)} ID đã tồn tại, {len(new_ids)} ID mới")
            
            # Xóa các bản ghi cũ nếu có
            if existing_ids:
                print(f"Đang xóa {len(existing_ids)} bản ghi cũ...")
                self.delete(existing_ids)
                time.sleep(0.5)  # Đợi xóa hoàn tất
            
            # Embedding tất cả texts cùng lúc theo batch
            print("Bắt đầu embedding texts...")
            vectors = self._batch_embed_texts(texts, batch_size=150)  # Batch nhỏ hơn cho embedding
            
            # Xử lý insert theo batch
            all_inserted_ids = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                print(f"Đang insert batch {i//batch_size + 1}/{total_batches} ({len(batch_texts)} items)")
                
                # Chuẩn bị dữ liệu cho batch hiện tại
                batch_data = []
                for j, (text, metadata, vector, image_id) in enumerate(zip(batch_texts, batch_metadatas, batch_vectors, batch_ids)):
                    record = {
                        "id": image_id,
                        "vector": vector,
                        "text": text,
                        "image_path": metadata.get("image_path", ""),
                        "category": metadata.get("category", ""),
                        "style": metadata.get("style", ""),
                        "app_name": metadata.get("app_name", ""),
                        "metadata": metadata
                    }
                    batch_data.append(record)
                
                # Insert batch vào collection
                try:
                    insert_result = self.vectorstore.collection.insert(batch_data)
                    all_inserted_ids.extend(batch_ids)
                    
                    # Flush sau mỗi batch
                    self.vectorstore.collection.flush()
                    print(f"Đã insert thành công batch {i//batch_size + 1}")
                    
                    # Nghỉ 1 giây trước khi xử lý batch tiếp theo
                    if i + batch_size < len(texts):
                        print("Nghỉ 1 giây trước batch tiếp theo...")
                        time.sleep(0.5)
                        
                except Exception as batch_error:
                    print(f"Lỗi khi insert batch {i//batch_size + 1}: {batch_error}")
                    # Thử insert từng item trong batch
                    for k, record in enumerate(batch_data):
                        try:
                            self.vectorstore.collection.insert([record])
                            all_inserted_ids.append(batch_ids[k])
                        except Exception as item_error:
                            print(f"Lỗi khi insert item {batch_ids[k]}: {item_error}")
                    
                    self.vectorstore.collection.flush()
            
            print(f"Hoàn thành! Đã xử lý {len(all_inserted_ids)}/{len(texts)} items")
            return all_inserted_ids
            
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
        Tìm kiếm similarity dựa trên text query
        
        Args:
            query: Text query để tìm kiếm
            k: Số lượng kết quả trả về
            category: Danh mục để lọc kết quả
            app_name: Tên app để lọc kết quả
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: Danh sách kết quả và score
        """
        # Tạo vector từ query
        query_vector = self.embedding_function.embed_query(query)
        
        # Tạo điều kiện tìm kiếm
        expr_conditions = []
        if category != "all":
            expr_conditions.append(f"category == '{category}'")
        if app_name != "all":
            expr_conditions.append(f"app_name == '{app_name}'")
        
        expr = None
        if expr_conditions:
            expr = " and ".join(expr_conditions)
        
        # Thực hiện tìm kiếm
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        print(f"Executing search with expr: {expr}")  # Debug log
        
        if expr:
            results = self.vectorstore.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=["id", "text", "image_path", "category", "style", "app_name", "metadata"]
            )
        else:
            results = self.vectorstore.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["id", "text", "image_path", "category", "style", "app_name", "metadata"]
            )
        

        # Format kết quả và tính toán điểm số
        docs_with_scores = []
        for hit in results[0]:
            doc = {
                "id": hit.entity.get("id"),
                "text": hit.entity.get("text"),
                "image_path": hit.entity.get("image_path"),
                "category": hit.entity.get("category"),
                "style": hit.entity.get("style", ""),
                "app_name": hit.entity.get("app_name", ""),
                "metadata": hit.entity.get("metadata", {})
            }
            
            # Tính điểm số với trọng số
            score = hit.score
            
            docs_with_scores.append((doc, score))
            
        return docs_with_scores
        
        raise

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