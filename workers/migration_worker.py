import asyncio
import threading
import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from aiohttp import ClientTimeout, ClientError
from services.image_service import embed_and_store_images
from util.image_processor import process_inventory_items_by_chunk, cleanup_session
from models.image_model import ImageMigrateProgressRequest

class MigrationWorker:
    """Worker để xử lý migration bất đồng bộ"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._running_jobs: Dict[str, bool] = {}  # Theo dõi trạng thái running của jobs
    
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
                "request": request.dict(),
                "retry_counts": {}  # Theo dõi số lần retry của từng chunk
            }
            self._running_jobs[job_id] = True
        
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

    async def _run_migration_async(self, job_id: str, request: ImageMigrateProgressRequest):
        """Phiên bản async của _run_migration với xử lý timeout tốt hơn"""
        try:
            self._update_job_status(job_id, {
                "status": "running",
                "message": "Đang bắt đầu migration..."
            })
            
            total_processed = 0
            total_errors = 0
            current_offset = request.start_offset
            chunk_count = 0
            
            while self._running_jobs.get(job_id, False):
                chunk_count += 1
                chunk_key = f"chunk_{chunk_count}"
                retry_count = self.jobs[job_id]["retry_counts"].get(chunk_key, 0)
                
                try:
                    # Xử lý chunk với timeout
                    async with asyncio.timeout(300):  # timeout 5 phút cho mỗi chunk
                        print(f"[Job {job_id}] Bắt đầu xử lý chunk {chunk_count}")
                        
                        chunk_result = await process_inventory_items_by_chunk(
                            chunk_size=request.chunk_size,
                            start_offset=current_offset,
                            api_batch_size=1000,
                            max_retries=3,
                            recreate_collection=request.recreate_collection
                        )
                        
                        if not chunk_result["data"]:
                            print(f"[Job {job_id}] Chunk {chunk_count}: Không có dữ liệu")
                            break
                        
                        # Xử lý embedding và lưu trữ
                        embed_result = await self._embed_and_store_with_cancel_check(
                            job_id,
                            chunk_result["data"],
                            request.recreate_collection if chunk_count == 1 else False,
                            request.batch_size
                        )
                        
                        if embed_result is None:  # Job bị cancel
                            print(f"[Job {job_id}] Chunk {chunk_count}: Job bị hủy")
                            return
                        
                        # Cập nhật thống kê
                        total_processed += embed_result["processed_count"]
                        total_errors += embed_result.get("error_count", 0)
                        
                        # Cập nhật progress
                        self._update_chunk_progress(
                            job_id, 
                            chunk_count,
                            total_processed,
                            total_errors,
                            chunk_result
                        )
                        
                        # Cập nhật offset và kiểm tra điều kiện dừng
                        current_offset = chunk_result["next_offset"]
                        if not chunk_result["has_more"]:
                            print(f"[Job {job_id}] Đã xử lý hết dữ liệu")
                            break
                        
                        # Delay giữa các chunk
                        await asyncio.sleep(2)
                        
                except asyncio.TimeoutError:
                    retry_count += 1
                    self.jobs[job_id]["retry_counts"][chunk_key] = retry_count
                    
                    if retry_count >= 3:  # Giới hạn số lần retry cho mỗi chunk
                        error_msg = f"Chunk {chunk_count} thất bại sau 3 lần thử do timeout"
                        self._log_chunk_error(job_id, chunk_count, error_msg)
                        total_errors += 1
                        continue
                    
                    wait_time = 2 ** retry_count  # exponential backoff
                    print(f"[Job {job_id}] Chunk {chunk_count} timeout, thử lại lần {retry_count} sau {wait_time}s")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    error_msg = f"Lỗi khi xử lý chunk {chunk_count}: {str(e)}"
                    self._log_chunk_error(job_id, chunk_count, error_msg)
                    total_errors += 1
                    continue
            
            # Cleanup và hoàn thành
            await cleanup_session()
            self._complete_migration(job_id, total_processed, total_errors)
            
        except Exception as e:
            self._handle_migration_error(job_id, e)
        finally:
            self._running_jobs.pop(job_id, None)
    
    def _run_migration(self, job_id: str, request: ImageMigrateProgressRequest):
        """Wrapper để chạy async code trong thread"""
        asyncio.run(self._run_migration_async(job_id, request))
    
    def _update_chunk_progress(
        self, 
        job_id: str, 
        chunk_count: int,
        total_processed: int,
        total_errors: int,
        chunk_result: Dict
    ):
        """Cập nhật tiến độ sau khi xử lý chunk thành công"""
        progress = (total_processed / chunk_result["total_records"] * 100) if chunk_result["total_records"] > 0 else 0
        
        self._update_job_status(job_id, {
            "total_processed": total_processed,
            "total_errors": total_errors,
            "progress": progress,
            "current_chunk": chunk_count,
            "total_records": chunk_result["total_records"],
            "message": f"Đã xử lý {total_processed}/{chunk_result['total_records']} items ({progress:.1f}%)"
        })
    
    def _log_chunk_error(self, job_id: str, chunk_count: int, error_msg: str):
        """Log lỗi xử lý chunk"""
        print(f"[Job {job_id}] {error_msg}")
        self._update_job_status(job_id, {
            "message": f"Chunk {chunk_count}: {error_msg}"
        })
    
    def _complete_migration(self, job_id: str, total_processed: int, total_errors: int):
        """Đánh dấu migration hoàn thành"""
        self._update_job_status(job_id, {
            "status": "completed",
            "progress": 100,
            "end_time": datetime.now(),
            "message": f"Hoàn thành! Đã xử lý {total_processed} items, {total_errors} lỗi"
        })
    
    def _handle_migration_error(self, job_id: str, error: Exception):
        """Xử lý lỗi trong quá trình migration"""
        error_msg = f"Lỗi trong quá trình migrate: {str(error)}"
        print(f"[Job {job_id}] {error_msg}")
        import traceback
        traceback.print_exc()
        
        self._update_job_status(job_id, {
            "status": "failed",
            "end_time": datetime.now(),
            "error": error_msg,
            "message": error_msg
        })
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Hủy job đang chạy
        
        Args:
            job_id: ID của job
            
        Returns:
            bool: True nếu thành công
        """
        with self.lock:
            if job_id in self.jobs:
                self._running_jobs[job_id] = False
                self.jobs[job_id]["status"] = "cancelled"
                self.jobs[job_id]["message"] = "Job đã bị hủy"
                self.jobs[job_id]["end_time"] = datetime.now()
                return True
            return False
    
    def _update_job_status(self, job_id: str, updates: Dict[str, Any]):
        """Cập nhật trạng thái job thread-safe"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)
    
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
        from util.env_loader import get_env
        from util.image_processor import update_sync_status
        
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
            
            # Cập nhật trạng thái đồng bộ sau mỗi 5000 records
            if len(processed_ids) > 0:
                # Chia thành các batch 5000 records để cập nhật
                batch_size_sync = 5000
                for i in range(0, len(processed_ids), batch_size_sync):
                    batch_ids = [metadatas[j]["id"] for j in range(i, min(i + batch_size_sync, len(processed_ids)))]
                    
                    # Kiểm tra cancel trước khi cập nhật
                    job_status = self.get_job_status(job_id)
                    if job_status and job_status["status"] == "cancelled":
                        return None
                    
                    try:
                        success = await update_sync_status(batch_ids)
                        if success:
                            print(f"[Job {job_id}] Đã cập nhật trạng thái đồng bộ cho {len(batch_ids)} records")
                        else:
                            print(f"[Job {job_id}] Lỗi khi cập nhật trạng thái đồng bộ cho batch {i//batch_size_sync + 1}")
                    except Exception as e:
                        print(f"[Job {job_id}] Lỗi khi cập nhật trạng thái đồng bộ: {str(e)}")
                        # Tiếp tục với batch tiếp theo ngay cả khi có lỗi
                    
                    # Delay nhỏ giữa các lần cập nhật
                    await asyncio.sleep(0.5)
            
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