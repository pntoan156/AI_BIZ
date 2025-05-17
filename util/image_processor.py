import os
import csv
import base64
import aiohttp
import io
from typing import List, Dict, Optional

async def fetch_products_csv(api_url: str) -> List[Dict]:
    """
    Lấy và xử lý file products.csv từ API
    
    Args:
        api_url (str): URL của API get-products-csv
        
    Returns:
        List[Dict]: Danh sách thông tin sản phẩm từ CSV
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch CSV: {response.status}")
            
            # Đọc và parse CSV từ response
            csv_text = await response.text()
            csv_file = io.StringIO(csv_text)
            reader = csv.DictReader(csv_file)
            return list(reader)

async def fetch_images(api_url: str, image_ids: List[str]) -> List[Dict]:
    """
    Lấy dữ liệu ảnh từ API theo batch
    
    Args:
        api_url (str): URL của API get-images
        image_ids (List[str]): Danh sách ID ảnh cần lấy
        
    Returns:
        List[Dict]: Danh sách thông tin ảnh
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json={"image_ids": image_ids}) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch images: {response.status}")
            
            result = await response.json()
            if not result.get("success"):
                raise Exception(f"API error: {result.get('message')}")
                
            return result.get("data", [])

async def process_products_in_batches(batch_size: int = 500):
    """
    Xử lý sản phẩm theo batch
    
    Args:
        products_api_url (str): URL của API get-products-csv
        images_api_url (str): URL của API get-images
        batch_size (int): Kích thước mỗi batch
    
    Returns:
        List[Dict]: Danh sách dữ liệu đã xử lý
    """
    # Example URLs - sẽ được cấu hình sau
    PRODUCTS_API_URL = "http://example.com/api/get-products-csv"
    IMAGES_API_URL = "http://example.com/api/get-images"
    
    # Lấy danh sách sản phẩm từ CSV
    products = await fetch_products_csv(PRODUCTS_API_URL)
    if not products:
        raise Exception("Không tìm thấy sản phẩm nào trong CSV")
    
    # Xử lý theo batch
    all_processed_data = []
    for i in range(0, len(products), batch_size):
        batch = products[i:i + batch_size]
        image_ids = [p['id'] for p in batch]
        
        # Lấy dữ liệu ảnh cho batch hiện tại
        images_data = await fetch_images(IMAGES_API_URL, image_ids)
        
        # Kết hợp thông tin từ CSV với dữ liệu ảnh
        for img_data in images_data:
            product_info = next((p for p in batch if p['id'] == img_data['image_id']), None)
            if product_info:
                img_data['image_name'] = product_info.get('name', '')
        
        all_processed_data.extend(images_data)
        print(f"Đã xử lý batch {i//batch_size + 1}/{(len(products) + batch_size - 1)//batch_size}")
    
    return all_processed_data