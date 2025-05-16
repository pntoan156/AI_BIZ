"""
Module di chuyển dữ liệu sản phẩm từ API vào vector store
"""
from typing import Dict, Any, List, Optional
import aiohttp
import csv
import os
from fastapi import HTTPException
from pydantic import BaseModel

# region: models
class MigrateRequest(BaseModel):
    """
    Schema cho request migrate sản phẩm
    """
    db_id: str
    inventory_ids: Optional[List[str]] = None

class MigrateResponse(BaseModel):
    """
    Schema cho response migrate sản phẩm
    """
    success: bool
    message: str
    total_migrated: int

# endregion

def load_products_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Đọc dữ liệu sản phẩm từ file CSV
    
    Args:
        csv_path: Đường dẫn tới file CSV
        
    Returns:
        List[Dict[str, Any]]: Danh sách sản phẩm
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV không tồn tại: {csv_path}")
        
    products = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(dict(row))
            
    return products