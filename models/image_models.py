from pydantic import BaseModel
from typing import Optional, List

class ImageModel(BaseModel):
    id: str
    name: str
    type: str
    unit_name: str
    file_names: str
    category: str

class ImageEmbedRequest(BaseModel):
    recreate_collection: bool = False
    pass

class ImageEmbedResponse(BaseModel):
    success: bool
    message: str
    total_processed: int = 0
    total_error: int = 0 
    
class ImageSearchRequest(BaseModel):
    product_names: List[str]
    category: Optional[str] = "all"  # Mặc định là "all" nếu không truyền
    
class ProductImageResult(BaseModel):
    product_name: str
    image_path: str
    category: str
    weight: float  # Thêm trường weight để lưu trọng số của kết quả

class ImageSearchResponse(BaseModel):
    results: List[ProductImageResult] 
