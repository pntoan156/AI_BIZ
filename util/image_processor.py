import os
import csv
import base64
import aiohttp
import io
from typing import List, Dict, Optional
from util.env_loader import get_env
from models.image_models import ImageModel

async def fetch_inventory_items(api_url: str, page_index: int = 1, page_size: int = 500) -> Dict[str, List[ImageModel]]:
    """
    Lấy danh sách sản phẩm từ API với phân trang
    
    Args:
        api_url (str): URL của API get-inventory-items
        page_index (int): Số trang
        page_size (int): Số lượng bản ghi mỗi trang
        
    Returns:
        Dict: {
            "Data": List[ImageModel] - Danh sách sản phẩm,
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
        async with session.post(api_url, headers=headers, json=params) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch inventory items: {response.status}")
            
            data = await response.json()
            
            # Chuyển đổi dữ liệu thành list ImageModel
            if data.get("Data"):
                data["Data"] = [ImageModel(**item) for item in data["Data"]]
            
            return data

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
    
    # Xử lý tất cả các trang
    all_processed_data = []
    for page in range(1, total_pages + 1):
        # Lấy dữ liệu trang hiện tại
        page_data = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, page, batch_size)
        
        # Xử lý dữ liệu từng sản phẩm trong trang
        for item in page_data["Data"]:
            # Tạo image_name bao gồm tên, đơn vị tính và style để embedding tốt hơn
            image_name = f"{item.name} ({item.unit_name})"
            if item.style and item.style.strip():
                image_name += f" {item.style}"
            
            processed_item = {
                'image_id': item.id,
                'image_path': item.file_names,
                'image_name': image_name,
                'category': item.category,
                'style': item.style,
                'app_name': item.app_name
            }
            all_processed_data.append(processed_item)
            
        print(f"Đã xử lý trang {page}/{total_pages}")
    
    return all_processed_data

async def process_inventory_items_by_chunk(
    chunk_size: int = 20000, 
    start_offset: int = 0,
    api_batch_size: int = 500
) -> Dict[str, any]:
    """
    Xử lý sản phẩm theo chunk lớn (ví dụ 20,000 items mỗi lần)
    
    Args:
        chunk_size (int): Số lượng sản phẩm mỗi chunk
        start_offset (int): Vị trí bắt đầu (để tiếp tục từ lần trước)
        api_batch_size (int): Kích thước batch khi gọi API
    
    Returns:
        Dict: {
            "data": List[Dict] - Dữ liệu đã xử lý,
            "total_records": int - Tổng số bản ghi,
            "processed_count": int - Số lượng đã xử lý trong chunk này,
            "next_offset": int - Offset cho chunk tiếp theo,
            "has_more": bool - Còn dữ liệu để xử lý không
        }
    """
    INVENTORY_ITEMS_API_URL = get_env("INVENTORY_ITEMS_API_URL")
    
    # Lấy trang đầu tiên để biết tổng số bản ghi
    first_page = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, 1, api_batch_size)
    if not first_page.get("Data"):
        raise Exception("Không tìm thấy sản phẩm nào")
    
    total_records = first_page["TotalRecords"]
    
    # Tính toán trang bắt đầu và kết thúc cho chunk này
    start_page = (start_offset // api_batch_size) + 1
    end_offset = min(start_offset + chunk_size, total_records)
    end_page = (end_offset - 1) // api_batch_size + 1
    
    print(f"Xử lý chunk từ offset {start_offset} đến {end_offset}")
    print(f"Trang từ {start_page} đến {end_page} (tổng {total_records} records)")
    
    # Xử lý các trang trong chunk
    chunk_processed_data = []
    items_processed_in_chunk = 0
    
    for page in range(start_page, end_page + 1):
        # Lấy dữ liệu trang hiện tại
        page_data = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, page, api_batch_size)
        
        # Xử lý dữ liệu từng sản phẩm trong trang
        for i, item in enumerate(page_data["Data"]):
            # Tính vị trí global của item này
            global_index = (page - 1) * api_batch_size + i
            
            # Bỏ qua nếu chưa đến offset bắt đầu
            if global_index < start_offset:
                continue
                
            # Dừng nếu đã vượt quá chunk size
            if items_processed_in_chunk >= chunk_size:
                break
            
            # Tạo image_name bao gồm tên, đơn vị tính và style để embedding tốt hơn
            image_name = f"{item.name} ({item.unit_name})"
            if item.style and item.style.strip():
                image_name += f" {item.style}"
            
            processed_item = {
                'image_id': item.id,
                'image_path': item.file_names,
                'image_name': image_name,
                'category': item.category,
                'style': item.style,
                'app_name': item.app_name
            }
            chunk_processed_data.append(processed_item)
            items_processed_in_chunk += 1
            
        print(f"Đã xử lý trang {page}/{end_page} - {items_processed_in_chunk} items trong chunk")
        
        # Dừng nếu đã đủ chunk size
        if items_processed_in_chunk >= chunk_size:
            break
    
    next_offset = start_offset + items_processed_in_chunk
    has_more = next_offset < total_records
    
    return {
        "data": chunk_processed_data,
        "total_records": total_records,
        "processed_count": items_processed_in_chunk,
        "next_offset": next_offset,
        "has_more": has_more,
        "current_offset": start_offset
    }