from fastapi import FastAPI
import os
from dotenv import load_dotenv
from migrate_products_to_store import MigrateRequest, MigrateResponse
from services.image_tags_service import search_images_by_name
from util.image_processor import process_inventory_items_in_batches
from stores.image_tags_store import embed_and_store_images
from models.image_tags_models import ImageSearchRequest, ImageSearchResponse, ImageTagsEmbedRequest, ImageTagsEmbedResponse
from models.image_tags_models import (
    ImageTagsEmbedRequest,
    ImageTagsEmbedResponse,
    ImageSearchRequest,
    ImageSearchResponse
)

load_dotenv()

app = FastAPI()

@app.post("/api/v1/images/migrate", response_model=ImageTagsEmbedResponse)
async def migrate_images(request: ImageTagsEmbedRequest):
    """
    Endpoint di chuyển batch ảnh vào vector store
    
    Args:
        request (ImageTagsEmbedRequest): Thông tin request
        
    Returns:
        ImageTagsEmbedResponse: Phản hồi
    """
    try:
        # Xử lý ảnh từ API bên ngoài
        images_data = await process_inventory_items_in_batches()
        
        if not images_data:
            return ImageTagsEmbedResponse(
                success=False,
                message="Không tìm thấy ảnh hợp lệ để xử lý",
                total_processed=0,
                total_error=0
            )
        
        # Embed và lưu trữ
        result = await embed_and_store_images(images_data, request.recreate_collection)
        
        return ImageTagsEmbedResponse(
            success=result["success"],
            message=f"Đã xử lý {result['processed_count']} ảnh, {result['error_count']} lỗi",
            total_processed=result["processed_count"],
            total_error=result["error_count"]
        )
        
    except Exception as e:
        return ImageTagsEmbedResponse(
            success=False,
            message=f"Lỗi: {str(e)}",
            total_processed=0,
            total_error=0
        )

@app.post("/api/v1/images/search-by-name", response_model=ImageSearchResponse)
async def search_images(request: ImageSearchRequest):
    """
    Endpoint tìm kiếm ảnh theo nhiều tên sản phẩm, trả về ảnh có độ tương đồng cao nhất cho mỗi sản phẩm
    """
    results = []
    for product_name in request.product_names:
        result = await search_images_by_name(
            image_name=product_name,
            limit=1
        )
        if result.get("results"):
            best_match = result["results"][0]
            results.append({
                "product_name": product_name,
                "image_path": best_match.image_path
            })
    
    return ImageSearchResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    # sync_tools_to_store()
    uvicorn.run(app, host="0.0.0.0", port=8500)