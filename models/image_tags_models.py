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
    image_name: str
    limit: int = 10
    
class ImageSearchResult(BaseModel):
    id: str
    name: str
    image_path: str
    score: float

class ImageSearchResponse(BaseModel):
    success: bool
    message: str
    results: List[ImageSearchResult]
    total: int 
