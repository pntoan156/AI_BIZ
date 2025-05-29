import asyncio
import threading
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from services.image_service import embed_and_store_images
from util.image_processor import process_inventory_items_by_chunk
from models.image_model import ImageMigrateProgressRequest

class MigrationWorker:
    """Worker để xử lý migration bất đồng bộ"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def start_migration_job(self, request: ImageMigrateProgressRequest) -> str:
        """
        Bắt đầu job migration mới
        
        Args:
            request: Thông tin request migration
            
        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "status": "starting",
                "progress": 0,
                "total_processed": 0,
                "total_errors": 0,
                "total_records": 0,
                "current_chunk": 0,
                "total_chunks": 0,
                "next_offset": request.start_offset,
                "has_more": True,
                "message": "Đang khởi tạo job...",
                "start_time": datetime.now(),
                "end_time": None,
                "error": None,
                "request": request.dict()
            }
        
        # Chạy migration trong thread riêng
        thread = threading.Thread(
            target=self._run_migration,
            args=(job_id, request),
            daemon=True
        )
        thread.start()
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy trạng thái job
        
        Args:
            job_id: ID của job
            
        Returns:
            Dict: Thông tin trạng thái job
        """
        with self.lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Lấy tất cả jobs"""
        with self.lock:
            return self.jobs.copy()
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Hủy job (đánh dấu để dừng)
        
        Args:
            job_id: ID của job
            
        Returns:
            bool: True nếu thành công
        """
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = "cancelled"
                self.jobs[job_id]["message"] = "Job đã bị hủy"
                return True
            return False
    
    def _update_job_status(self, job_id: str, updates: Dict[str, Any]):
        """Cập nhật trạng thái job thread-safe"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)
    
    def _run_migration(self, job_id: str, request: ImageMigrateProgressRequest):
        """
        Chạy migration trong thread riêng
        
        Args:
            job_id: ID của job
            request: Thông tin request
        """
        try:
            # Cập nhật trạng thái bắt đầu
            self._update_job_status(job_id, {
                "status": "running",
                "message": "Đang bắt đầu migration..."
            })
            
            total_processed = 0
            total_errors = 0
            total_records = 0
            current_offset = request.start_offset
            chunk_count = 0
            total_chunks = 0
            
            print(f"[Job {job_id}] Bắt đầu migrate từ offset {current_offset}")
            
            while True:
                # Kiểm tra nếu job bị hủy
                job_status = self.get_job_status(job_id)
                if job_status and job_status["status"] == "cancelled":
                    print(f"[Job {job_id}] Job bị hủy, dừng xử lý")
                    return
                
                chunk_count += 1
                print(f"[Job {job_id}] === CHUNK {chunk_count} - Offset: {current_offset} ===")
                
                # Kiểm tra giới hạn chunks
                if request.max_chunks > 0 and chunk_count > request.max_chunks:
                    print(f"[Job {job_id}] Đã đạt giới hạn {request.max_chunks} chunks")
                    break
                
                # Cập nhật trạng thái
                self._update_job_status(job_id, {
                    "message": f"Đang xử lý chunk {chunk_count}...",
                    "current_chunk": chunk_count
                })
                
                # Kiểm tra cancel trước khi fetch data
                job_status = self.get_job_status(job_id)
                if job_status and job_status["status"] == "cancelled":
                    print(f"[Job {job_id}] Job bị hủy trong quá trình fetch data")
                    return
                
                # Xử lý chunk hiện tại
                chunk_result = asyncio.run(process_inventory_items_by_chunk(
                    chunk_size=request.chunk_size,
                    start_offset=current_offset,
                    api_batch_size=500
                ))
                
                # Cập nhật total_records từ chunk đầu tiên
                if chunk_count == 1:
                    total_records = chunk_result["total_records"]
                    total_chunks = (total_records + request.chunk_size - 1) // request.chunk_size
                    if request.max_chunks > 0:
                        total_chunks = min(total_chunks, request.max_chunks)
                    
                    self._update_job_status(job_id, {
                        "total_records": total_records,
                        "total_chunks": total_chunks
                    })
                    
                    print(f"[Job {job_id}] Tổng số bản ghi: {total_records}, Ước tính {total_chunks} chunks")
                
                # Kiểm tra nếu không còn dữ liệu
                if not chunk_result["data"]:
                    print(f"[Job {job_id}] Chunk {chunk_count}: Không có dữ liệu, kết thúc")
                    break
                
                print(f"[Job {job_id}] Chunk {chunk_count}: Lấy được {len(chunk_result['data'])} items")
                
                # Kiểm tra cancel trước khi embed
                job_status = self.get_job_status(job_id)
                if job_status and job_status["status"] == "cancelled":
                    print(f"[Job {job_id}] Job bị hủy trước khi embed")
                    return
                
                # Embed và lưu trữ chunk hiện tại với cancel checking
                embed_result = asyncio.run(self._embed_and_store_with_cancel_check(
                    job_id,
                    chunk_result["data"], 
                    request.recreate_collection if chunk_count == 1 else False,
                    request.batch_size
                ))
                
                # Kiểm tra nếu job bị cancel trong quá trình embed
                if embed_result is None:
                    print(f"[Job {job_id}] Job bị hủy trong quá trình embed")
                    return
                
                # Cập nhật thống kê
                total_processed += embed_result["processed_count"]
                total_errors += embed_result["error_count"]
                
                # Tính phần trăm tiến độ
                progress_percentage = (total_processed / total_records * 100) if total_records > 0 else 0
                
                # Cập nhật trạng thái
                self._update_job_status(job_id, {
                    "total_processed": total_processed,
                    "total_errors": total_errors,
                    "progress": progress_percentage,
                    "current_chunk": chunk_count,
                    "message": f"Đã xử lý {total_processed}/{total_records} items ({progress_percentage:.1f}%)"
                })
                
                print(f"[Job {job_id}] Chunk {chunk_count}: Đã xử lý {embed_result['processed_count']}/{len(chunk_result['data'])} items")
                print(f"[Job {job_id}] Tổng tiến độ: {total_processed}/{total_records} ({progress_percentage:.1f}%)")
                
                # Cập nhật offset cho chunk tiếp theo
                current_offset = chunk_result["next_offset"]
                
                # Kiểm tra nếu đã hết dữ liệu
                if not chunk_result["has_more"]:
                    print(f"[Job {job_id}] Hoàn thành! Đã xử lý hết {chunk_count} chunks")
                    break
                
                # Kiểm tra cancel trước khi nghỉ
                job_status = self.get_job_status(job_id)
                if job_status and job_status["status"] == "cancelled":
                    print(f"[Job {job_id}] Job bị hủy trước khi nghỉ")
                    return
                
                # Nghỉ 2 giây giữa các chunk
                print(f"[Job {job_id}] Nghỉ 2 giây trước chunk tiếp theo...")
                time.sleep(2)
            
            # Hoàn thành thành công
            self._update_job_status(job_id, {
                "status": "completed",
                "progress": 100,
                "end_time": datetime.now(),
                "next_offset": current_offset,
                "has_more": chunk_result.get("has_more", False) if 'chunk_result' in locals() else False,
                "message": f"Hoàn thành! Đã xử lý {chunk_count} chunks, {total_processed} items thành công, {total_errors} lỗi"
            })
            
            print(f"[Job {job_id}] Migration hoàn thành thành công")
            
        except Exception as e:
            # Xử lý lỗi
            import traceback
            error_msg = f"Lỗi trong quá trình migrate: {str(e)}"
            print(f"[Job {job_id}] {error_msg}")
            traceback.print_exc()
            
            self._update_job_status(job_id, {
                "status": "failed",
                "end_time": datetime.now(),
                "error": error_msg,
                "message": error_msg
            })
    
    async def _embed_and_store_with_cancel_check(
        self, 
        job_id: str,
        images_data: list, 
        recreate_collection: bool = False,
        batch_size: int = 20000  # Tăng batch_size cho mega insert
    ):
        """
        Embed và store với kiểm tra cancel status - SỬ DỤNG TRỰC TIẾP mega_insert_texts
        """
        from stores.image_store import ImageStore
        
        # Kiểm tra cancel trước khi bắt đầu
        job_status = self.get_job_status(job_id)
        if job_status and job_status["status"] == "cancelled":
            return None
        
        total_records = len(images_data)
        print(f"[Job {job_id}] CÁCH 1: Mega insert trực tiếp {total_records:,} records với batch_size={batch_size:,}")
        
        # Auto-optimize batch size
        if total_records < 5000:
            batch_size = min(batch_size, 2000)
        elif total_records > 50000:
            batch_size = min(batch_size, 30000)
        
        # Khởi tạo image store
        image_store = ImageStore(recreate_collection=recreate_collection)
        
        try:
            # Chuẩn bị dữ liệu một lần duy nhất - tối ưu
            texts = [img_data['image_name'] for img_data in images_data]
            metadatas = [{
                "id": img_data['image_id'],
                "image_path": img_data['image_path'],
                "image_name": img_data['image_name'],
                "category": img_data['category'],
                "style": img_data['style'],
                "app_name": img_data['app_name']
            } for img_data in images_data]
            
            # Kiểm tra cancel sau khi prep
            job_status = self.get_job_status(job_id)
            if job_status and job_status["status"] == "cancelled":
                return None
            
            print(f"[Job {job_id}] mega_insert_texts - tối ưu")
            
            # SỬ DỤNG TRỰC TIẾP mega_insert_texts - KHÔNG chia nhỏ
            processed_ids = image_store.vectorstore.mega_insert_texts(
                texts=texts,
                metadatas=metadatas,
                batch_size=batch_size
            )
            
            # Kiểm tra cancel sau khi hoàn thành
            job_status = self.get_job_status(job_id)
            if job_status and job_status["status"] == "cancelled":
                return None
            
            print(f"[Job {job_id}] hoàn thành: {len(processed_ids):,}/{total_records:,} records")
            
            return {
                "success": True,
                "processed_count": len(processed_ids),
                "error_count": total_records - len(processed_ids),
                "total": total_records,
                "method": "mega_insert_direct_optimized",
                "batch_size": batch_size
            }
            
        except Exception as e:
            print(f"[Job {job_id}] Error trong CÁCH 1: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "processed_count": 0,
                "error_count": total_records,
                "total": total_records,
                "method": "mega_insert_direct_failed"
            }

# Singleton instance
migration_worker = MigrationWorker() 