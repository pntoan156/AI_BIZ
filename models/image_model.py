from pydantic import BaseModel
from typing import Optional, List

class ImageModel(BaseModel):
    id: str
    name: str
    type: str
    unit_name: str
    file_names: str
    category: str
    style: Optional[str] = ""
    app_name: Optional[str] = ""

class ImageSearchRequest(BaseModel):
    product_names: List[str]
    category: Optional[str] = "all"  # Mặc định là "all" nếu không truyền
    app_name: Optional[str] = "all"  # Mặc định là "all" nếu không truyền
    
class ProductImageResult(BaseModel):
    product_name: str
    image_path: str
    category: str
    style: str
    app_name: str
    weight: float  # Thêm trường weight để lưu trọng số của kết quả

class ImageSearchResponse(BaseModel):
    results: List[ProductImageResult] 

class ImageMigrateProgressRequest(BaseModel):
    recreate_collection: bool = False
    batch_size: int = 5000  # Giảm batch size xuống 50
    chunk_size: int = 20000
    start_offset: int = 0
    max_chunks: int = 0  # 0 = không giới hạn, xử lý hết tất cả

class ImageMigrateProgressResponse(BaseModel):
    success: bool
    message: str
    total_processed: int = 0
    total_error: int = 0
    total_records: int = 0
    current_chunk: int = 0
    total_chunks: int = 0
    progress_percentage: float = 0.0
    next_offset: int = 0
    has_more: bool = False
    processing_time_seconds: float = 0.0 
