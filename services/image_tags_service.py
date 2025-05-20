import json
import os
from typing import Dict, Any, Optional, List, Tuple
from stores.image_tags_store import ImageTagsStore
from models.image_tags_models import ProductImageResult

async def process_tags_file(
    folder_path: str,
    tags_file_path: Optional[str] = None,
    db_id: str = None,
    recreate_collection: bool = False
) -> Dict[str, Any]:
    """
    Xử lý file tags và lưu vào vector store
    
    Args:
        folder_path: Đường dẫn thư mục chứa ảnh
        tags_file_path: Đường dẫn file tags_data.json
        db_id: ID của database
        recreate_collection: Có tạo mới collection không
        
    Returns:
        Dict[str, Any]: Kết quả xử lý
    """
    try:
        # Nếu không cung cấp tags_file_path, tìm trong thư mục ảnh
        if not tags_file_path:
            tags_file_path = os.path.join(folder_path, "tags_data.json")
            
        # Kiểm tra tồn tại
        if not os.path.exists(tags_file_path):
            return {
                "success": False,
                "message": f"File tags {tags_file_path} không tồn tại"
            }
            
        # Đọc file tags
        with open(tags_file_path, "r", encoding="utf-8") as f:
            tags_data = json.load(f)
            
        if not tags_data:
            return {
                "success": False,
                "message": "File tags không có dữ liệu"
            }
            
        # Khởi tạo image tags store
        image_store = ImageTagsStore(recreate_collection=recreate_collection)
        
        # Chuẩn bị texts và metadata
        texts = []
        metadatas = []
        
        for tag_data in tags_data:
            texts.append(tag_data.get('name', ''))
            metadatas.append({
                "id": tag_data.get('id', ''),
                "image_path": tag_data.get('image_path', ''),
                "image_name": tag_data.get('name', ''),
            })
        
        # Thêm vào vector store
        image_store.add_texts(texts, metadatas)
        
        return {
            "success": True,
            "message": f"Đã xử lý {len(texts)} tags",
            "total_processed": len(texts),
            "total_error": 0
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Lỗi khi xử lý: {str(e)}",
            "total_processed": 0,
            "total_error": len(tags_data) if 'tags_data' in locals() else 0
        }

async def search_images_by_name(
    image_name: str,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Tìm kiếm ảnh theo tên sử dụng vector similarity
    
    Args:
        query: Tên ảnh cần tìm kiếm
        limit: Số lượng kết quả trả về
        db_id: ID của database để lọc kết quả
        
    Returns:
        Dict[str, Any]: Kết quả tìm kiếm
    """
    try:
        # Khởi tạo image store
        image_store = ImageTagsStore()
        
        # Thực hiện tìm kiếm vector similarity
        results = image_store.hybrid_search(image_name, limit)
            
        # Chuyển đổi kết quả sang định dạng response
        search_results = []
        for doc, score in results:
            result = ProductImageResult(
                product_name=doc.get("text", ""),  # text field contains the image name
                image_path=doc.get("image_path", "")
            )
            search_results.append(result)
            
        return {
            "success": True,
            "message": "Tìm kiếm thành công",
            "results": search_results,
            "total": len(search_results)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Lỗi khi tìm kiếm: {str(e)}",
            "results": [],
            "total": 0
        } 