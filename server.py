from fastapi import FastAPI
import time
from dotenv import load_dotenv
from services.image_service import search_images_by_name, embed_and_store_images
from util.image_processor import process_inventory_items_by_chunk
from models.image_model import (
    ImageSearchRequest, ImageSearchResponse,
    ImageMigrateProgressRequest, ImageMigrateProgressResponse
)

# Force load .env file và override system environment variables
load_dotenv(override=True)

app = FastAPI()

@app.post("/api/v1/images/migrate", response_model=ImageMigrateProgressResponse)
async def migrate_images(request: ImageMigrateProgressRequest):
    """
    Endpoint di chuyển ảnh vào vector store
    - Nếu max_chunks = 0: Xử lý TẤT CẢ dữ liệu tự động theo chunk
    - Nếu max_chunks > 0: Xử lý số chunk giới hạn
    - Hỗ trợ tiếp tục từ start_offset cụ thể
    
    Args:
        request (ImageMigrateProgressRequest): Thông tin request
        
    Returns:
        ImageMigrateProgressResponse: Phản hồi với thông tin tiến độ chi tiết
    """
    start_time = time.time()
    
    try:
        total_processed = 0
        total_errors = 0
        total_records = 0
        current_offset = request.start_offset
        chunk_count = 0
        total_chunks = 0
        
        print(f"Bắt đầu migrate từ offset {current_offset}")
        print(f"Chunk size: {request.chunk_size}, Batch size: {request.batch_size}")
        
        if request.max_chunks == 0:
            print("Chế độ: Xử lý TẤT CẢ dữ liệu")
        else:
            print(f"Chế độ: Xử lý tối đa {request.max_chunks} chunks")
        
        while True:
            chunk_count += 1
            print(f"\n=== CHUNK {chunk_count} - Offset: {current_offset} ===")
            
            # Kiểm tra giới hạn chunks
            if request.max_chunks > 0 and chunk_count > request.max_chunks:
                print(f"Đã đạt giới hạn {request.max_chunks} chunks, dừng xử lý")
                break
            
            # Xử lý chunk hiện tại
            chunk_result = await process_inventory_items_by_chunk(
                chunk_size=request.chunk_size,
                start_offset=current_offset,
                api_batch_size=500
            )
            
            # Cập nhật total_records và tính total_chunks từ chunk đầu tiên
            if chunk_count == 1:
                total_records = chunk_result["total_records"]
                total_chunks = (total_records + request.chunk_size - 1) // request.chunk_size
                if request.max_chunks > 0:
                    total_chunks = min(total_chunks, request.max_chunks)
                print(f"Tổng số bản ghi: {total_records}, Ước tính {total_chunks} chunks")
            
            # Kiểm tra nếu không còn dữ liệu
            if not chunk_result["data"]:
                print(f"Chunk {chunk_count}: Không có dữ liệu, kết thúc")
                break
            
            print(f"Chunk {chunk_count}: Lấy được {len(chunk_result['data'])} items")
            
            # Embed và lưu trữ chunk hiện tại
            embed_result = await embed_and_store_images(
                chunk_result["data"], 
                request.recreate_collection if chunk_count == 1 else False,  # Chỉ recreate ở chunk đầu
                request.batch_size
            )
            
            # Cập nhật thống kê
            total_processed += embed_result["processed_count"]
            total_errors += embed_result["error_count"]
            
            # Tính phần trăm tiến độ
            progress_percentage = (total_processed / total_records * 100) if total_records > 0 else 0
            
            print(f"Chunk {chunk_count}: Đã xử lý {embed_result['processed_count']}/{len(chunk_result['data'])} items")
            print(f"Tổng tiến độ: {total_processed}/{total_records} ({progress_percentage:.1f}%)")
            
            # Cập nhật offset cho chunk tiếp theo
            current_offset = chunk_result["next_offset"]
            
            # Kiểm tra nếu đã hết dữ liệu
            if not chunk_result["has_more"]:
                print(f"\nHoàn thành! Đã xử lý hết {chunk_count} chunks")
                break
            
            # Nghỉ 2 giây giữa các chunk để tránh quá tải
            print("Nghỉ 2 giây trước chunk tiếp theo...")
            time.sleep(2)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return ImageMigrateProgressResponse(
            success=True,
            message=f"Hoàn thành migrate! Đã xử lý {chunk_count} chunks, {total_processed} items thành công, {total_errors} lỗi trong {processing_time:.1f}s",
            total_processed=total_processed,
            total_error=total_errors,
            total_records=total_records,
            current_chunk=chunk_count,
            total_chunks=total_chunks,
            progress_percentage=(total_processed / total_records * 100) if total_records > 0 else 100,
            next_offset=current_offset,
            has_more=chunk_result.get("has_more", False) if 'chunk_result' in locals() else False,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        import traceback
        traceback.print_exc()
        return ImageMigrateProgressResponse(
            success=False,
            message=f"Lỗi trong quá trình migrate: {str(e)}",
            total_processed=total_processed,
            total_error=total_errors,
            total_records=total_records,
            current_chunk=chunk_count,
            total_chunks=total_chunks,
            progress_percentage=(total_processed / total_records * 100) if total_records > 0 else 0,
            next_offset=current_offset,
            has_more=True,
            processing_time_seconds=processing_time
        )

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
            filtered_results = [r for r in result.results if r.weight >= 0.4]
            if filtered_results:
                # Lấy kết quả tốt nhất từ các kết quả đã lọc
                best_match = filtered_results[0]
                # Giữ nguyên tên sản phẩm được tìm kiếm thay vì tên trong kho
                best_match.product_name = product_name
                results.append(best_match)
    
    return ImageSearchResponse(results=results)

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

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)