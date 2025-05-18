import os
import csv
import base64
import aiohttp
import io
from typing import List, Dict, Optional

async def fetch_inventory_items_csv(api_url: str) -> List[Dict]:
    """
    Lấy và xử lý file inventory_items.csv từ API
    
    Args:
        api_url (str): URL của API get-inventory-items-csv
        
    Returns:
        List[Dict]: Danh sách thông tin sản phẩm từ CSV
    """
    headers = {
        "x-ai-key": "AI-MISA-ESHOP-2025-310526be-637a-40dd-b50d-ed8a677fd9e2"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch CSV: {response.status}")
            
            # Đọc và parse CSV từ response
            csv_text = await response.text()
            csv_file = io.StringIO(csv_text)
            reader = csv.DictReader(csv_file)
            return list(reader)

async def process_inventory_items_in_batches(batch_size: int = 500):
    """
    Xử lý sản phẩm theo batch
    
    Args:
        inventory_items_api_url (str): URL của API get-inventory-items-csv
        batch_size (int): Kích thước mỗi batch
    
    Returns:
        List[Dict]: Danh sách dữ liệu đã xử lý
    """
    # Example URLs - sẽ được cấu hình sau
    INVENTORY_ITEMS_API_URL = "https://eshopapp.misa.vn/g2/api/aibiz/aibizs/inventory-items-info"
    
    # Lấy danh sách sản phẩm từ CSV
    inventory_items = await fetch_inventory_items_csv(INVENTORY_ITEMS_API_URL)
    if not inventory_items:
        raise Exception("Không tìm thấy sản phẩm nào trong CSV")
    
    # Xử lý theo batch
    all_processed_data = []
    for i in range(0, len(inventory_items), batch_size):
        batch = inventory_items[i:i + batch_size]
        
        # Xử lý dữ liệu từ CSV
        for item in batch:
            processed_item = {
                'image_id': item['image_id'],
                'image_path': item.get('image_path', ''),
                'image_name': item.get('image_name', '')
            }
            all_processed_data.append(processed_item)
            
        print(f"Đã xử lý batch {i//batch_size + 1}/{(len(inventory_items) + batch_size - 1)//batch_size}")
    
    return all_processed_data