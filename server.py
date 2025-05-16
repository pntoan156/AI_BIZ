from fastapi import FastAPI
import os
from dotenv import load_dotenv
from migrate_products_to_store import MigrateRequest, MigrateResponse, load_products_from_csv
from services.image_tags_service import process_tags_file, search_images_by_name
from util.image_processor import process_images_folder
from stores.image_tags_store import embed_and_store_images
from models.image_tags_models import (
    ImageTagsEmbedRequest,
    ImageTagsEmbedResponse,
    ImageSearchRequest,
    ImageSearchResponse
)

load_dotenv()

app = FastAPI()

@app.post("/api/v1/images/migrate-folder", response_model=ImageTagsEmbedResponse)
async def migrate_images_folder(request: ImageTagsEmbedRequest):
    """
    Endpoint di chuyển ảnh từ thư mục vào vector store
    
    Args:
        request (MigrateRequest): Thông tin request
        
    Returns:
        MigrateResponse: Phản hồi
    """
    try:
        # Xử lý ảnh trong thư mục
        images_data = await process_images_folder(
            request.folder_path, 
            None,  # Không cần tags file nữa
            request.db_id
        )
        
        if not images_data:
            return MigrateResponse(
                success=False,
                message="Không tìm thấy ảnh hợp lệ trong thư mục",
                total_migrated=0
            )
        
        # Embed và lưu trữ
        result = await embed_and_store_images(images_data, request.recreate_collection)
        
        return MigrateResponse(
            success=result["success"],
            message=f"Đã xử lý {result['processed_count']} ảnh, {result['error_count']} lỗi",
            total_migrated=result["processed_count"]
        )
        
    except Exception as e:
        return MigrateResponse(
            success=False,
            message=f"Lỗi: {str(e)}",
            total_migrated=0
        )

@app.post("/api/v1/images/search-by-name", response_model=ImageSearchResponse)
async def search_images(request: ImageSearchRequest):
    """
    Endpoint tìm kiếm ảnh theo tên sử dụng vector similarity
    
    Args:
        request (ImageSearchRequest): Thông tin request tìm kiếm
        
    Returns:
        ImageSearchResponse: Kết quả tìm kiếm
    """
    result = await search_images_by_name(
        image_name=request.image_name,
        limit=request.limit
    )
    
    return ImageSearchResponse(**result)

if __name__ == "__main__":
    import uvicorn
    # sync_tools_to_store()
    uvicorn.run(app, host="0.0.0.0", port=8000)