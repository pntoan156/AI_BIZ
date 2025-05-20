from pydantic import BaseModel
from typing import Optional, List

class ImageTagsEmbedRequest(BaseModel):
    db_id: str
    recreate_collection: bool = False

class ImageTagsEmbedResponse(BaseModel):
    success: bool
    message: str
    total_processed: int = 0
    total_error: int = 0 
    
class ImageSearchRequest(BaseModel):
    product_names: List[str]
    
class ProductImageResult(BaseModel):
    product_name: str
    image_path: str

class ImageSearchResponse(BaseModel):
    results: List[ProductImageResult] 
