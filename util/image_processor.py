import os
import csv
import base64
import aiohttp
import io
from typing import List, Dict, Optional
from util.env_loader import get_env

async def fetch_inventory_items(api_url: str, page_index: int = 1, page_size: int = 500) -> Dict:
    """
    Lấy danh sách sản phẩm từ API với phân trang
    
    Args:
        api_url (str): URL của API get-inventory-items
        page_index (int): Số trang
        page_size (int): Số lượng bản ghi mỗi trang
        
    Returns:
        Dict: {
            "Data": List[Dict] - Danh sách sản phẩm (id, name, type, unit_name, file_names),
            "TotalRecords": int - Tổng số bản ghi
        }
    """
    headers = {
        "x-ai-key": "AI-MISA-ESHOP-2025-310526be-637a-40dd-b50d-ed8a677fd9e2"
    }
    
    params = {
        "pageIndex": page_index,
        "pageSize": page_size
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url, headers=headers, params=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch inventory items: {response.status}")
            
            return await response.json()

async def process_inventory_items_in_batches(batch_size: int = 500):
    """
    Xử lý sản phẩm theo batch
    
    Args:
        batch_size (int): Kích thước mỗi batch
    
    Returns:
        List[Dict]: Danh sách dữ liệu đã xử lý
    """
    INVENTORY_ITEMS_API_URL = get_env("INVENTORY_ITEMS_API_URL")
    
    # Lấy trang đầu tiên để biết tổng số bản ghi
    first_page = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, 1, batch_size)
    if not first_page.get("Data"):
        raise Exception("Không tìm thấy sản phẩm nào")
    
    total_records = first_page["TotalRecords"]
    total_pages = (total_records + batch_size - 1) // batch_size
    
    # Xử lý dữ liệu từ trang đầu
    all_processed_data = []
    for item in first_page["Data"]:
        processed_item = {
            'image_id': item['id'],
            'image_path': item.get('file_names', ''),
            'image_name': f"{item.get('name', '')} ({item.get('unit_name', '')})"
        }
        all_processed_data.append(processed_item)
    
    # Xử lý các trang tiếp theo
    for page in range(2, total_pages + 1):
        page_data = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, page, batch_size)
        
        for item in page_data["Data"]:
            processed_item = {
                'image_id': item['id'],
                'image_path': item.get('file_names', ''),
                'image_name': f"{item.get('name', '')} ({item.get('unit_name', '')})"
            }
            all_processed_data.append(processed_item)
            
        print(f"Đã xử lý trang {page}/{total_pages}")
    
    return all_processed_data