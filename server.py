from fastapi import FastAPI
import os
from dotenv import load_dotenv
from services.image_service import search_images_by_name
from util.image_processor import process_inventory_items_in_batches
from services.image_service import embed_and_store_images
from models.image_models import ImageEmbedRequest, ImageEmbedResponse, ImageSearchRequest, ImageSearchResponse

load_dotenv()

app = FastAPI()

@app.post("/api/v1/images/migrate", response_model=ImageEmbedResponse)
async def migrate_images(request: ImageEmbedRequest):
    """
    Endpoint di chuyển batch ảnh vào vector store
    
    Args:
        request (ImageEmbedRequest): Thông tin request
        
    Returns:
        ImageEmbedResponse: Phản hồi
    """
    try:
        # Xử lý ảnh từ API bên ngoài
        images_data = await process_inventory_items_in_batches()
        
        if not images_data:
            return ImageEmbedResponse(
                success=False,
                message="Không tìm thấy ảnh hợp lệ để xử lý",
                total_processed=0,
                total_error=0
            )
        
        # Embed và lưu trữ
        result = await embed_and_store_images(images_data, request.recreate_collection)
        
        return ImageEmbedResponse(
            success=result["success"],
            message=f"Đã xử lý {result['processed_count']} ảnh, {result['error_count']} lỗi",
            total_processed=result["processed_count"],
            total_error=result["error_count"]
        )
        
    except Exception as e:
        return ImageEmbedResponse(
            success=False,
            message=f"Lỗi: {str(e)}",
            total_processed=0,
            total_error=0
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
            filtered_results = [r for r in result.results if r.weight >= 0]
            if filtered_results:
                # Lấy kết quả tốt nhất từ các kết quả đã lọc
                best_match = filtered_results[0]
                results.append(best_match)
    
    return ImageSearchResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)