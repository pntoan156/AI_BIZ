"""
Module xử lý việc đọc biến môi trường từ file .env
"""
import os
from typing import Optional

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Đọc giá trị biến môi trường và bỏ qua comment
    
    Args:
        key: Tên biến môi trường
        default: Giá trị mặc định nếu không tìm thấy
        
    Returns:
        Giá trị của biến môi trường hoặc default
    """
    value = os.getenv(key, default)
    if value is None:
        return default
        
    # Tìm vị trí comment (#)
    comment_pos = value.find('#')
    if comment_pos != -1:
        # Lấy phần trước comment và loại bỏ khoảng trắng
        value = value[:comment_pos].strip()
        
    return value 