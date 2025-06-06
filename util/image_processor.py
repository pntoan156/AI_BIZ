import os
import csv
import base64
import aiohttp
import io
import asyncio
from typing import List, Dict, Optional
from aiohttp import ClientTimeout, ClientError
from tenacity import (
    retry,
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)
from util.env_loader import get_env
from models.image_model import ImageModel

# Biến global để lưu session
_session: Optional[aiohttp.ClientSession] = None

async def get_session() -> aiohttp.ClientSession:
    """
    Lấy hoặc tạo mới session
    """
    global _session
    if _session is None or _session.closed:
        timeout = ClientTimeout(
            total=30,        # Tổng thời gian tối đa cho request
            connect=10,      # Thời gian tối đa để kết nối
            sock_read=20     # Thời gian tối đa để đọc dữ liệu
        )
        _session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=10,  # Giới hạn số connection đồng thời
                enable_cleanup_closed=True
            )
        )
    return _session

async def cleanup_session():
    """
    Đóng session khi cần
    """
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None

@retry(
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    stop=stop_after_attempt(3),  # Thử lại tối đa 3 lần
    wait=wait_exponential(multiplier=1, min=4, max=10)  # Chờ 4-10 giây giữa các lần retry
)
async def fetch_inventory_items(
    api_url: str, 
    page_index: int = 1, 
    page_size: int = 500,
    recreate_collection: bool = False
) -> Dict[str, List[ImageModel]]:
    """
    Lấy danh sách sản phẩm từ API với phân trang và xử lý timeout
    
    Args:
        api_url (str): URL của API get-inventory-items
        page_index (int): Số trang
        page_size (int): Số lượng bản ghi mỗi trang
        recreate_collection (bool): Nếu True thì onlyNotSynced=False, ngược lại True
        
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
        "pageSize": page_size,
        "onlyNotSynced": not recreate_collection  # Đảo ngược logic: True -> False và False -> True
    }
    
    try:
        session = await get_session()
        async with session.post(api_url, headers=headers, json=params) as response:
            if response.status != 200:
                raise aiohttp.ClientError(f"Failed to fetch inventory items: {response.status}")
            
            data = await response.json()
            
            # Thêm delay nhỏ để tránh quá tải server
            await asyncio.sleep(0.5)
            
            # Chuyển đổi dữ liệu thành list ImageModel
            if data.get("Data"):
                data["Data"] = [ImageModel(**item) for item in data["Data"]]
            
            return data
            
    except asyncio.TimeoutError as e:
        print(f"Timeout khi gọi API {api_url} - page {page_index}: {str(e)}")
        raise  # Raise lại để retry mechanism xử lý
        
    except aiohttp.ClientError as e:
        print(f"Lỗi kết nối khi gọi API {api_url} - page {page_index}: {str(e)}")
        raise  # Raise lại để retry mechanism xử lý
        
    except Exception as e:
        print(f"Lỗi không mong đợi khi gọi API {api_url} - page {page_index}: {str(e)}")
        raise  # Các lỗi khác sẽ không được retry

async def process_inventory_items_in_batches(batch_size: int = 500, recreate_collection: bool = False):
    """
    Xử lý sản phẩm theo batch
    
    Args:
        batch_size (int): Kích thước mỗi batch
        recreate_collection (bool): Nếu True thì onlyNotSynced=True, ngược lại False
    
    Returns:
        List[Dict]: Danh sách dữ liệu đã xử lý
    """
    INVENTORY_ITEMS_API_URL = get_env("INVENTORY_ITEMS_API_URL")
    
    # Lấy trang đầu tiên để biết tổng số bản ghi
    first_page = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, 1, batch_size, recreate_collection)
    if not first_page.get("Data"):
        raise Exception("Không tìm thấy sản phẩm nào")
    
    total_records = first_page["TotalRecords"]
    total_pages = (total_records + batch_size - 1) // batch_size
    
    # Xử lý tất cả các trang
    all_processed_data = []
    for page in range(1, total_pages + 1):
        # Lấy dữ liệu trang hiện tại
        page_data = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, page, batch_size, recreate_collection)
        
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
    api_batch_size: int = 500,
    max_retries: int = 3,
    recreate_collection: bool = False
) -> Dict[str, any]:
    """
    Xử lý sản phẩm theo chunk lớn với retry mechanism
    
    Args:
        chunk_size (int): Số lượng sản phẩm mỗi chunk
        start_offset (int): Vị trí bắt đầu
        api_batch_size (int): Kích thước batch khi gọi API
        max_retries (int): Số lần thử lại tối đa cho mỗi request
        recreate_collection (bool): Nếu True thì onlyNotSynced=True, ngược lại False
        
    Returns:
        Dict: {
            "data": List[Dict] - Dữ liệu đã xử lý,
            "total_records": int - Tổng số bản ghi,
            "processed_count": int - Số lượng đã xử lý trong chunk này,
            "next_offset": int - Offset cho chunk tiếp theo,
            "has_more": bool - Còn dữ liệu để xử lý không,
            "current_offset": int - Offset hiện tại
        }
    """
    INVENTORY_ITEMS_API_URL = get_env("INVENTORY_ITEMS_API_URL")
    
    try:
        # Lấy trang đầu tiên để biết tổng số bản ghi
        first_page = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, 1, api_batch_size, recreate_collection)
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
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Lấy dữ liệu trang với timeout
                    async with asyncio.timeout(60):  # timeout 60s cho mỗi request
                        page_data = await fetch_inventory_items(INVENTORY_ITEMS_API_URL, page, api_batch_size, recreate_collection)
                        
                        # Xử lý dữ liệu từng sản phẩm trong trang
                        for i, item in enumerate(page_data["Data"]):
                            # Tính vị trí global của item
                            global_index = (page - 1) * api_batch_size + i
                            
                            if global_index < start_offset:
                                continue
                                
                            if items_processed_in_chunk >= chunk_size:
                                break
                            
                            # Xử lý item
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
                        
                        # Thành công, thoát khỏi retry loop
                        break
                        
                except asyncio.TimeoutError:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Trang {page} thất bại sau {max_retries} lần thử")
                        break
                    
                    wait_time = 2 ** retry_count  # exponential backoff
                    print(f"Timeout trang {page}, thử lại lần {retry_count} sau {wait_time}s")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý trang {page}: {str(e)}")
                    break
            
            # Kiểm tra nếu đã đủ số lượng
            if items_processed_in_chunk >= chunk_size:
                break
            
            # Delay giữa các trang để tránh overload
            await asyncio.sleep(0.5)
        
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
        
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý chunk: {str(e)}")
        raise
    finally:
        # Cleanup session nếu cần
        if not has_more:
            await cleanup_session()

@retry(
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def update_sync_status(image_ids: List[str]) -> bool:
    """
    Cập nhật trạng thái đồng bộ cho danh sách sản phẩm
    
    Args:
        image_ids (List[str]): Danh sách ID sản phẩm cần cập nhật
        
    Returns:
        bool: True nếu cập nhật thành công
    """
    api_url = get_env("UPDATE_SYNC_STATUS_URL")
    headers = {
        "x-ai-key": "AI-MISA-ESHOP-2025-310526be-637a-40dd-b50d-ed8a677fd9e2"
    }
    
    try:
        session = await get_session()
        async with session.post(
            api_url, 
            headers=headers, 
            json=image_ids  # Truyền trực tiếp mảng image_ids
        ) as response:
            status = response.status
            print(f"Response status: {status}")
            
            if status != 200:
                response_text = await response.text()
                print(f"Error response: {response_text}")
                print(f"Lỗi khi cập nhật trạng thái đồng bộ: {status}")
                return False
            
            print(f"Cập nhật thành công {len(image_ids)} records")
            print(f"=========================\n")
            return True
            
    except asyncio.TimeoutError as e:
        print(f"Timeout khi cập nhật trạng thái đồng bộ: {str(e)}")
        print(f"=========================\n")
        raise
        
    except aiohttp.ClientError as e:
        print(f"Lỗi kết nối khi cập nhật trạng thái đồng bộ: {str(e)}")
        print(f"=========================\n")
        raise
        
    except Exception as e:
        print(f"Lỗi không mong đợi khi cập nhật trạng thái đồng bộ: {str(e)}")
        print(f"=========================\n")
        raise