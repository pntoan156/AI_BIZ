from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import time
from dotenv import load_dotenv
from services.image_service import search_images_by_name, embed_and_store_images, upsert_image, get_image_by_id
from util.image_processor import process_inventory_items_by_chunk
from workers.migration_worker import migration_worker
from models.image_model import (
    ImageSearchRequest, ImageSearchResponse,
    ImageMigrateProgressRequest, ImageMigrateProgressResponse,
    ImageModel
)
from util.env_loader import get_env

# Force load .env file và override system environment variables
load_dotenv(override=True)

app = FastAPI()

@app.post("/api/v1/images/migrate")
async def migrate_images(request: ImageMigrateProgressRequest):
    """
    Endpoint di chuyển ảnh vào vector store (bất đồng bộ với worker)
    
    Args:
        request (ImageMigrateProgressRequest): Thông tin request
        
    Returns:
        Dict: Job ID để theo dõi tiến độ
    """
    try:
        job_id = migration_worker.start_migration_job(request)
        return {
            "success": True,
            "job_id": job_id,
            "message": "Đã bắt đầu migration job. Sử dụng job_id để theo dõi tiến độ."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/v1/images/migrate-status/{job_id}")
async def get_migration_status(job_id: str):
    """
    Lấy trạng thái migration job
    
    Args:
        job_id: ID của job
        
    Returns:
        Dict: Thông tin trạng thái job
    """
    job_status = migration_worker.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    # Tạo copy để không thay đổi dữ liệu gốc
    job_status_copy = job_status.copy()
    
    # Format datetime cho JSON serialization
    from datetime import datetime
    if job_status_copy.get("start_time") and isinstance(job_status_copy["start_time"], datetime):
        job_status_copy["start_time"] = job_status_copy["start_time"].isoformat()
    if job_status_copy.get("end_time") and isinstance(job_status_copy["end_time"], datetime):
        job_status_copy["end_time"] = job_status_copy["end_time"].isoformat()
    
    return {
        "success": True,
        "data": job_status_copy
    }

@app.get("/api/v1/images/migrate-jobs")
async def get_all_migration_jobs():
    """
    Lấy tất cả migration jobs
    
    Returns:
        Dict: Danh sách tất cả jobs
    """
    jobs = migration_worker.get_all_jobs()
    
    # Tạo copy để không thay đổi dữ liệu gốc
    jobs_copy = {}
    from datetime import datetime
    
    for job_id, job_data in jobs.items():
        job_copy = job_data.copy()
        # Format datetime cho JSON serialization
        if job_copy.get("start_time") and isinstance(job_copy["start_time"], datetime):
            job_copy["start_time"] = job_copy["start_time"].isoformat()
        if job_copy.get("end_time") and isinstance(job_copy["end_time"], datetime):
            job_copy["end_time"] = job_copy["end_time"].isoformat()
        jobs_copy[job_id] = job_copy
    
    return {
        "success": True,
        "data": jobs_copy
    }

@app.post("/api/v1/images/migrate-cancel/{job_id}")
async def cancel_migration_job(job_id: str):
    """
    Hủy migration job
    
    Args:
        job_id: ID của job
        
    Returns:
        Dict: Kết quả hủy job
    """
    success = migration_worker.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    return {
        "success": True,
        "message": "Đã hủy job thành công"
    }

@app.post("/api/v1/images/search-by-name", response_model=ImageSearchResponse)
async def search_images(request: ImageSearchRequest):
    """
    Endpoint tìm kiếm ảnh theo nhiều tên sản phẩm, trả về ảnh có độ tương đồng cao nhất cho mỗi sản phẩm
    và có trọng số > 0.4
    """
    results = []
    for product_name in request.product_names:
        result = await search_images_by_name(
            image_name=product_name,
            category=request.category,
            app_name=request.app_name
        )
        if result.results:
            # Lọc kết quả có trọng số > 0.4
            weight_threshold = float(get_env("WEIGHT_THRESHOLD"))
            filtered_results = [r for r in result.results if r.weight >= weight_threshold]
            if filtered_results:
                # Lấy kết quả tốt nhất từ các kết quả đã lọc
                best_match = filtered_results[0]
                # Giữ nguyên tên sản phẩm được tìm kiếm thay vì tên trong kho
                best_match.product_name = product_name
                results.append(best_match)
    
    return ImageSearchResponse(results=results)

@app.post("/api/v1/images/upsert")
async def upsert_image_endpoint(image_model: ImageModel):
    """
    Endpoint upsert (thêm mới hoặc cập nhật) một ImageModel trong Milvus
    
    Args:
        image_model (ImageModel): Thông tin image cần upsert
        
    Returns:
        Dict: Kết quả upsert với thông tin chi tiết
    """
    try:
        result = await upsert_image(image_model)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Image {result['action']} thành công",
                "data": {
                    "action": result["action"],
                    "image_id": result["image_id"],
                    "image_name": result["image_name"],
                    "is_update": result.get("is_update", False),
                    "timing": result["timing"],
                    "metadata": result["metadata"]
                }
            }
        else:
            return {
                "success": False,
                "error": result["error"],
                "image_id": result.get("image_id", "unknown")
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Lỗi server khi upsert image"
        })

@app.get("/api/v1/images/collection-info")
async def get_collection_info():
    """
    Endpoint lấy thông tin về collection
    
    Returns:
        Dict: Thông tin collection
    """
    try:
        from stores.image_store import ImageStore
        image_store = ImageStore()
        info = image_store.get_collection_info()
        return {
            "success": True,
            "data": info
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/v1/images/{image_id}")
async def get_image_endpoint(image_id: str):
    """
    Endpoint lấy thông tin image theo ID
    
    Args:
        image_id (str): ID của image
        
    Returns:
        Dict: Thông tin image
    """
    try:
        result = await get_image_by_id(image_id)
        
        if result["success"]:
            if result["found"]:
                return {
                    "success": True,
                    "data": result["data"]
                }
            else:
                return {
                    "success": False,
                    "error": "Not found",
                    "message": result["message"]
                }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Lỗi server khi lấy thông tin image"
        })

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)