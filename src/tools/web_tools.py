"""
Web Tools - Web search, page browsing, and image search.

Provides asynchronous tools for performing Google web searches (via Serper API),
fetching and extracting webpage content (via Jina Reader API), and downloading
images from Google Image Search.

Environment variables:
    SERPER_API_KEY: API key for Google Serper (https://serper.dev).
    JINA_API_KEY:   API key for Jina Reader (https://jina.ai).
    IMAGE_SAVE_DIR: Directory for saving downloaded images (default: ./cache/searched_images).
    WEB_CACHE_DIR:  Directory for the web-request cache (default: ./cache).
"""
import os
import json
import re
import asyncio
import aiohttp
import random  # noqa: F401 – used by RateLimiter
import time
import chardet
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all configurable via environment variables)
# ---------------------------------------------------------------------------
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

IMAGE_SAVE_DIR = os.getenv("IMAGE_SAVE_DIR", os.path.join(".", "cache", "searched_images"))
CACHE_DIR = os.getenv("WEB_CACHE_DIR", os.path.join(".", "cache"))
WEB_CACHE_FILE = os.path.join(CACHE_DIR, "web_cache.json")

# Error indicators for web content
ERROR_INDICATORS = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]


class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, rate_limit: int, time_window: int = 60):
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            while self.tokens <= 0:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.rate_limit,
                    self.tokens + (time_passed * self.rate_limit / self.time_window)
                )
                self.last_update = now
                if self.tokens <= 0:
                    await asyncio.sleep(random.randint(2, 10))
            self.tokens -= 1
            return True


class WebCache:
    """Cache for web search and page browsing results"""
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self._lock = asyncio.Lock()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
        return {"web_search": {}, "page_browser": {}}

    def _save_cache(self):
        # Deprecated: usage moved to save_final_cache to avoid race conditions
        pass

    def save_final_cache(self):
        try:
            # Load latest cache from disk to ensure we don't overwrite other processes' updates
            latest_cache = self._load_cache()
            
            # Merge local in-memory cache into the latest cache
            # We prioritize our local updates/additions
            if "web_search" in self.cache:
                if "web_search" not in latest_cache:
                    latest_cache["web_search"] = {}
                latest_cache["web_search"].update(self.cache["web_search"])
                
            if "page_browser" in self.cache:
                if "page_browser" not in latest_cache:
                    latest_cache["page_browser"] = {}
                latest_cache["page_browser"].update(self.cache["page_browser"])
            
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(latest_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved and merged web cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    async def get_search(self, query: str) -> Optional[List[Dict]]:
        async with self._lock:
            return self.cache.get("web_search", {}).get(query)

    async def set_search(self, query: str, results: List[Dict]):
        async with self._lock:
            if "web_search" not in self.cache:
                self.cache["web_search"] = {}
            self.cache["web_search"][query] = results
            # self._save_cache()  # Don't save immediately

    async def get_page(self, url: str) -> Optional[str]:
        async with self._lock:
            return self.cache.get("page_browser", {}).get(url)

    async def set_page(self, url: str, content: str):
        async with self._lock:
            if "page_browser" not in self.cache:
                self.cache["page_browser"] = {}
            self.cache["page_browser"][url] = content
            # self._save_cache()  # Don't save immediately


# Global cache instance
_web_cache = WebCache(WEB_CACHE_FILE)

def save_web_cache():
    """Save the accumulated web cache to disk, merging with existing file."""
    _web_cache.save_final_cache()

# Global rate limiters
_serper_rate_limiter = RateLimiter(rate_limit=100)
_jina_rate_limiter = RateLimiter(rate_limit=100)


async def _extract_text_from_url_async(
    url: str,
    session: aiohttp.ClientSession,
    use_jina: bool = True,
    timeout: int = 30
) -> Tuple[str, str]:
    """
    Extract text content from a URL asynchronously.
    
    Returns:
        Tuple of (extracted_text, full_text)
    """
    try:
        if use_jina:
            await _jina_rate_limiter.acquire()
            jina_headers = {
                'Authorization': f'Bearer {JINA_API_KEY}',
                'X-Return-Format': 'markdown',
            }
            async with session.get(
                f'https://r.jina.ai/{url}',
                headers=jina_headers,
                ssl=False,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                text = await response.text()
                # Clean up markdown formatting
                pattern = r"\(https?:.*?\)|\[https?:.*?\]"
                text = re.sub(pattern, "", text)
                text = text.replace('---', '-').replace('===', '=').replace('   ', ' ')
                return text[:10000], text
        else:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                content_type = response.headers.get('content-type', '').lower()
                if 'charset' in content_type:
                    charset = content_type.split('charset=')[-1]
                    html = await response.text(encoding=charset)
                else:
                    content = await response.read()
                    detected = chardet.detect(content)
                    encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
                    html = content.decode(encoding, errors='replace')
                
                # Check for error indicators
                has_error = (
                    any(ind.lower() in html.lower() for ind in ERROR_INDICATORS) and len(html.split()) < 64
                ) or len(html) < 50
                
                if has_error:
                    return "Error: Content contains error indicators", ""
                
                try:
                    soup = BeautifulSoup(html, 'lxml')
                except Exception:
                    soup = BeautifulSoup(html, 'html.parser')
                
                text = soup.get_text(separator=' ', strip=True)
                return text[:10000], text
                
    except Exception as e:
        error_msg = f"Error fetching {url}: {str(e)}"
        return error_msg, error_msg


async def web_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a web search using Google Serper API and return search results (title, url, snippet).
    Does NOT fetch full page content.
    
    Args:
        query: The search query string
        top_k: Maximum number of results to return (default: 10)
        
    Returns:
        List of search results with title, url, snippet, and date.
    """
    # Check cache first
    cached_results = await _web_cache.get_search(query)
    if cached_results:
        return cached_results[:top_k]

    await _serper_rate_limiter.acquire()
    
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": top_k})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    for attempt in range(5):
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, data=payload, ssl=False) as response:
                    response.raise_for_status()
                    search_results = await response.json()

            # Extract organic results
            results = []
            if 'organic' in search_results:
                for i, result in enumerate(search_results['organic'][:top_k]):
                    results.append({
                        'id': i + 1,
                        'title': result.get('title', ''),
                        'url': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'date': result.get('date', ''),
                    })

            # Save to cache
            await _web_cache.set_search(query, results)

            return results

        except Exception as e:
            logger.error(f"Web search failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return [{"error": f"Search failed: {str(e)}"}]


async def page_browser(urls: List[str]) -> Dict[str, str]:
    """
    Fetch the content of multiple webpages.
    
    Args:
        urls: A list of URLs to fetch content from
        
    Returns:
        Dictionary mapping URLs to their fetched content
    """
    # Handle string input (sometimes passed as single string instead of list)
    if isinstance(urls, str):
        urls_str = urls.strip()
        # Check if it's a stringified list
        if urls_str.startswith('[') and urls_str.endswith(']'):
            try:
                parsed = json.loads(urls_str)
                if isinstance(parsed, list):
                    urls = parsed
                else:
                    urls = [urls]
            except json.JSONDecodeError:
                urls = [urls]
        else:
            urls = [urls]
            
    # Handle list input that might contain a single stringified list
    # (common artifact from LLM tool calling where string arg is wrapped in list)
    if isinstance(urls, list) and len(urls) == 1 and isinstance(urls[0], str):
        first_url = urls[0].strip()
        if first_url.startswith('[') and first_url.endswith(']'):
            try:
                parsed = json.loads(first_url)
                if isinstance(parsed, list):
                    urls = parsed
            except json.JSONDecodeError:
                pass
    
    results = {}
    urls_to_fetch = []

    # Check cache first
    for url in urls:
        cached_content = await _web_cache.get_page(url)
        if cached_content:
            results[url] = cached_content
        else:
            urls_to_fetch.append(url)
    
    if not urls_to_fetch:
        return results
    
    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                fetch_tasks = [
                    _extract_text_from_url_async(url, session, use_jina=True)
                    for url in urls_to_fetch
                ]
                contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)

                for url, content in zip(urls_to_fetch, contents):
                    if isinstance(content, Exception):
                        result_text = f"Error: {str(content)}"
                    else:
                        result_text = content[0] if content[0] else "No content extracted"

                    results[url] = result_text

                    # Save to cache only if successful (no Error prefix)
                    if not result_text.startswith("Error"):
                        await _web_cache.set_page(url, result_text)

            break
        except Exception as e:
            logger.error(f"Page browser failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            for url in urls_to_fetch:
                results[url] = f"Error: {str(e)}"
    
    return results


async def _download_image_async(
    session: aiohttp.ClientSession,
    image_url: str,
    save_path: str,
    timeout: int = 15
) -> bool:
    """Download an image asynchronously."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        async with session.get(
            image_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status == 200:
                content = await response.read()
                with open(save_path, 'wb') as f:
                    f.write(content)
                return True
    except Exception as e:
        logger.warning(f"Failed to download image {image_url}: {e}")
    return False


def _get_file_extension(url: str) -> str:
    """Extract file extension from URL."""
    parsed = urlparse(url)
    path = parsed.path
    if '.' in path:
        ext = '.' + path.split('.')[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            return ext
    return '.jpg'


async def web_image_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for images using Google Serper Image API and download them.
    
    Args:
        query: The search query for images
        top_k: Maximum number of images to return (default: 5)
        
    Returns:
        List of image information with local file paths
    """
    await _serper_rate_limiter.acquire()
    
    # Ensure save directory exists
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    url = "https://google.serper.dev/images"
    payload = json.dumps({"q": query, "num": top_k})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, data=payload, ssl=False) as response:
                response.raise_for_status()
                search_results = await response.json()
        
        results = []
        
        if 'images' in search_results:
            # Generate timestamp for unique filenames
            timestamp = time.strftime("%m%d%H%M")
            
            async with aiohttp.ClientSession() as session:
                download_tasks = []
                image_infos = []
                
                for i, img_info in enumerate(search_results['images'][:top_k]):
                    image_url = img_info.get('imageUrl', '')
                    title = img_info.get('title', f'image_{i}')
                    
                    # Generate safe filename
                    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:30]
                    safe_title = safe_title.replace(' ', '_')
                    random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
                    extension = _get_file_extension(image_url)
                    filename = f"{timestamp}_{random_suffix}_{safe_title}{extension}"
                    save_path = os.path.join(IMAGE_SAVE_DIR, filename)
                    
                    image_infos.append({
                        'title': title,
                        'source_url': image_url,
                        'local_path': save_path,
                        'source_page': img_info.get('link', ''),
                    })
                    
                    download_tasks.append(
                        _download_image_async(session, image_url, save_path)
                    )
                
                # Download all images concurrently
                download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
                
                for info, success in zip(image_infos, download_results):
                    if success is True:
                        results.append({
                            'title': info['title'],
                            'local_path': info['local_path'],
                            'source_page': info['source_page'],
                            'download_success': True,
                        })
                    else:
                        results.append({
                            'title': info['title'],
                            'local_path': None,
                            'source_url': info['source_url'],
                            'download_success': False,
                            'error': str(success) if isinstance(success, Exception) else 'Download failed',
                        })
        
        if not results:
            return [{"message": "No images found for the query."}]
        
        return results
        
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        return [{"error": f"Image search failed: {str(e)}"}]


# OpenAI Function Definitions

def get_openai_function_web_search() -> dict:
    """Return the OpenAI tool/function definition for web_search."""
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Perform a Google web search and return the top search results (title, URL, snippet, date). "
                "This tool does NOT fetch the full webpage content. "
                "Use this to find relevant URLs and summaries. "
                "If you need detailed content, use the page_browser tool with the URLs found here. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The search query string. Be specific and use keywords that are likely to return relevant results. "
                            "Example: 'Nobel Prize Physics 2023 winner' or 'population of Tokyo 2024'"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }


def get_openai_function_page_browser() -> dict:
    """Return the OpenAI tool/function definition for page_browser."""
    return {
        "type": "function",
        "function": {
            "name": "page_browser",
            "description": (
                "Fetch and extract the full text content from one or more webpages given their URLs. "
                "Use this when you have specific URLs (e.g., from search results) and need to read their detailed content. "
                "Returns a mapping from each URL to its extracted text content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string", "format": "uri"},
                        "description": (
                            "A list of webpage URLs to fetch content from. "
                            "Example: ['https://en.wikipedia.org/wiki/Tokyo', 'https://example.com/article']"
                        ),
                        "minItems": 1,
                        "maxItems": 10
                    }
                },
                "required": ["urls"]
            }
        }
    }


def get_openai_function_web_image_search() -> dict:
    """Return the OpenAI tool/function definition for web_image_search."""
    return {
        "type": "function",
        "function": {
            "name": "web_image_search",
            "description": (
                "Search for images on Google and download the top results. "
                "Use this when you need to find specific images from the internet that might contain valuable visual information. "
                "The downloaded images can then be analyzed using the visual_question_answering tool. "
                "Returns local file paths of successfully downloaded images."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The image search query. Be descriptive about what kind of image you're looking for. "
                            "Example: 'Eiffel Tower at night' or 'chemical structure of caffeine'"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }



if __name__ == "__main__":

    async def _test():
        print("Testing web_search...")
        results = await web_search("Python programming language history")
        print(json.dumps(results[:2], indent=2, ensure_ascii=False))

        print("\nTesting page_browser...")
        pages = await page_browser(["https://www.python.org"])
        for url, content in pages.items():
            print(f"{url}: {content[:500]}...")

        print("\nTesting web_image_search...")
        images = await web_image_search("sunset over ocean")
        print(json.dumps(images, indent=2, ensure_ascii=False))

    asyncio.run(_test())

