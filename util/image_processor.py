import os
import csv
from typing import List, Dict, Optional

async def process_images_folder(folder_path: str, tags_file_path: Optional[str], db_id: str):
    """
    Xử lý thư mục ảnh và lấy thông tin từ CSV
    
    Args:
        folder_path (str): Đường dẫn thư mục chứa cả images và products.csv
        tags_file_path (str): Đường dẫn file tags_data.json (không còn sử dụng)
        db_id (str): Database ID
        
    Returns:
        List: Danh sách dữ liệu đã xử lý
    """
    # Đọc thông tin từ CSV
    products = []
    csv_path = os.path.join(folder_path, 'products.csv')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        products = list(reader)
    
    # Quét thư mục ảnh
    images_data = []
    images_folder = os.path.join(folder_path, 'images')
    for filename in os.listdir(images_folder):
        if filename.lower().endswith('.jpg'):
            # Tìm thông tin sản phẩm từ CSV dựa trên file_name
            product_info = next((p for p in products if p.get('file_name') == filename), None)
            if not product_info:
                print(f"Warning: Không tìm thấy thông tin sản phẩm cho ảnh {filename}")
                continue
            
            # Đường dẫn đầy đủ tới file ảnh
            image_path = os.path.join(images_folder, filename)
            
            # Đọc dữ liệu ảnh
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
            
            # Thêm vào danh sách
            images_data.append({
                'image_id': product_info['id'],  # Sử dụng id từ CSV
                'image_name': product_info['name'],  # Sử dụng name từ CSV
                'image_path': product_info['id'] + '.jpg',
                'image_bytes': image_bytes,
                'database_id': db_id,
            })
    
    return images_data 