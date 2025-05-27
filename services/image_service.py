import json
import os
from typing import Dict, Any, Optional, List, Tuple
from stores.image_store import ImageStore
from models.image_model import ProductImageResult, ImageSearchResponse

async def search_images_by_name(
    image_name: str,
    category: str = "all",
    app_name: str = "all"
) -> ImageSearchResponse:
    """
    Tìm kiếm ảnh theo tên sử dụng vector similarity
    
    Args:
        image_name: Tên ảnh cần tìm
        category: Danh mục để lọc kết quả
        app_name: Tên app để lọc kết quả
        
    Returns:
        ImageSearchResponse: Kết quả tìm kiếm bao gồm danh sách kết quả với trọng số
    """
    try:
        # Khởi tạo image store
        image_store = ImageStore()
        
        # Thực hiện tìm kiếm vector similarity
        results = image_store.hybrid_search(image_name, 4, category=category, app_name=app_name)
            
        # Chuyển đổi kết quả sang định dạng response
        search_results = []
        for doc, score in results:
            result = ProductImageResult(
                product_name=doc.get("text", ""),  # Sử dụng tên thực tế từ database (text field)
                image_path=doc.get("image_path", ""),
                category=doc.get("category", ""),  # Lấy category trực tiếp từ doc
                style=doc.get("style", ""),
                app_name=doc.get("app_name", ""),
                weight=float(score)  # Chuyển đổi score thành float
            )
            search_results.append(result)
            
        return ImageSearchResponse(results=search_results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ImageSearchResponse(results=[])

async def embed_and_store_images(
    images_data: List[Dict], 
    recreate_collection: bool = False,
    batch_size: int = 50
):
    """
    Embed và lưu trữ ảnh vào vector store với batch processing
    
    Args:
        images_data (List[Dict]): Danh sách dữ liệu ảnh
        recreate_collection (bool): Có tạo lại collection không
        batch_size (int): Kích thước mỗi batch để xử lý
        
    Returns:
        Dict: Kết quả xử lý
    """
    # Khởi tạo image store
    image_store = ImageStore(recreate_collection=recreate_collection)
    
    try:
        # Chuẩn bị texts và metadata
        texts = []
        metadatas = []
        
        for img_data in images_data:
            texts.append(img_data['image_name'])
            metadatas.append({
                "id": img_data['image_id'],
                "image_path": img_data['image_path'],
                "image_name": img_data['image_name'],
                "category": img_data['category'],
                "style": img_data['style'],
                "app_name": img_data['app_name']
            })
        
        print(f"Bắt đầu xử lý {len(images_data)} items với batch size {batch_size}")
        
        # Sử dụng insert thường với batch size được chỉ định
        processed_ids = image_store.add_texts(texts, metadatas, batch_size=batch_size)
        
        return {
            "success": True,
            "processed_count": len(processed_ids),
            "error_count": len(images_data) - len(processed_ids),
            "total": len(images_data),
            "method": "regular_insert"
        }
        
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "processed_count": 0,
            "error_count": len(images_data),
            "total": len(images_data),
            "method": "failed"
        } 