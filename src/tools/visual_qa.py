"""
Visual Question Answering Tool - Interact with images / videos via an
OpenAI-compatible multimodal model (e.g. Qwen3-Omni).

Environment variables:
    VQA_BASE_URL:  Base URL of the VQA model endpoint (OpenAI-compatible).
    VQA_API_KEY:   API key for the VQA endpoint (default: "empty").
    VQA_MODEL:     Model name served at the endpoint.
"""
import os
import re
import json
import base64
import mimetypes
import asyncio
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all configurable via environment variables)
# ---------------------------------------------------------------------------
QWEN_BASE_URL = os.getenv("VQA_BASE_URL", "http://localhost:8000/v1")
QWEN_API_KEY = os.getenv("VQA_API_KEY", "empty")
QWEN_MODEL = os.getenv("VQA_MODEL", "qwen3-omni")

# Supported file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

# System prompt for VQA
VQA_SYSTEM_PROMPT = """You are a helpful visual assistant that can analyze images and videos to answer questions.
Provide accurate, detailed, and helpful responses based on what you observe in the visual content.
If you cannot determine something from the visual content, say so clearly.
Focus on providing factual observations rather than speculative interpretations."""


def _is_url(s: str) -> bool:
    """Check if string is a URL"""
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("data:"))


def _to_data_url(path: str, default_mime: str) -> str:
    """Convert a file path to a data URL"""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or default_mime
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.error(f"Failed to convert {path} to data URL: {e}")
        raise


def _detect_media_type(file_path: str) -> str:
    """Detect if file is image or video based on extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    else:
        # Try to guess from mime type
        mime, _ = mimetypes.guess_type(file_path)
        if mime:
            if mime.startswith('image/'):
                return "image"
            elif mime.startswith('video/'):
                return "video"
        # Default to image
        return "image"


def _build_media_part(media_type: str, source: str) -> dict:
    """Build a media content part for the API"""
    default_mimes = {
        "image": "image/jpeg",
        "video": "video/mp4",
    }
    
    if _is_url(source):
        url = source
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"{media_type} file not found: {source}")
        url = _to_data_url(source, default_mimes[media_type])
    
    return {
        "type": f"{media_type}_url",
        f"{media_type}_url": {"url": url},
    }


def _clean_response(content: str) -> str:
    """Clean the model response by removing think blocks"""
    # Remove <think>...</think> block
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


async def visual_question_answering(
    file_path: str,
    question: str,
    max_tokens: int = 20480,
    timeout: int = 1200
) -> Dict[str, Any]:
    """
    Ask a question about an image or video file.
    
    Args:
        file_path: Path to the image or video file (local path or URL)
        question: The question to ask about the visual content
        max_tokens: Maximum tokens in the response (default: 2048)
        timeout: Request timeout in seconds (default: 120)
        
    Returns:
        Dictionary containing:
        - answer: The model's response to the question
        - media_type: Whether the input was 'image' or 'video'
        - error: Error message if the request failed
    """
    result = {
        "answer": None,
        "media_type": None,
        "error": None,
    }
    
    try:
        # Detect media type
        media_type = _detect_media_type(file_path)
        result["media_type"] = media_type
        
        # Build content parts
        contents = []
        
        # Add media part
        try:
            media_part = _build_media_part(media_type, file_path)
            contents.append(media_part)
        except FileNotFoundError as e:
            result["error"] = str(e)
            return result
        except Exception as e:
            result["error"] = f"Failed to load media file: {str(e)}"
            return result
        
        # Add question text
        contents.append({"type": "text", "text": question})
        
        # Build messages
        messages = [
            {"role": "system", "content": VQA_SYSTEM_PROMPT},
            {"role": "user", "content": contents},
        ]
        
        # Create client and make request
        client = AsyncOpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)
        
        try:
            completion = await client.chat.completions.create(
                model=QWEN_MODEL,
                messages=messages,
                stream=False,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            
            response_content = completion.choices[0].message.content
            result["answer"] = _clean_response(response_content)
            
        finally:
            await client.close()
        
    except asyncio.TimeoutError:
        result["error"] = f"Request timed out after {timeout} seconds"
    except Exception as e:
        result["error"] = f"VQA request failed: {str(e)}"
        logger.error(f"VQA error for {file_path}: {e}")
    
    return result


def get_openai_function_visual_question_answering() -> dict:
    """Return the OpenAI tool/function definition for visual_question_answering."""
    return {
        "type": "function",
        "function": {
            "name": "visual_question_answering",
            "description": (
                "Ask a question about an image or video file and get an answer based on visual analysis. "
                "Use this tool to:\n"
                "1. Understand the content of images or video clips\n"
                "2. Extract specific visual information (text, objects, scenes, actions, etc.)\n"
                "3. Verify visual details for constructing accurate multi-hop questions\n"
                "4. Analyze downloaded images from web_image_search\n\n"
                "The tool automatically detects whether the input is an image or video based on file extension."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Path to the image or video file. Can be:\n"
                            "- Local file path: '/path/to/image.jpg' or '/path/to/video.mp4'\n"
                            "- URL: 'https://example.com/image.png'\n"
                            "Supported formats: JPG, PNG, GIF, WebP, BMP (images); MP4, AVI, MOV, MKV (videos)"
                        )
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "The question to ask about the visual content. Be specific about what you want to know. "
                            "Examples:\n"
                            "- 'What text is visible in this image?'\n"
                            "- 'Describe the main activity happening in this video'\n"
                            "- 'What objects are on the table?'\n"
                            "- 'What color is the car in the background?'"
                        )
                    }
                },
                "required": ["file_path", "question"]
            }
        }
    }



if __name__ == "__main__":

    async def _test():
        import sys

        print("Testing visual_question_answering...")
        if len(sys.argv) < 2:
            print("Usage: python visual_qa.py <image_path> [question]")
            return
        image_path = sys.argv[1]
        question = sys.argv[2] if len(sys.argv) > 2 else "Describe what you see in this image."
        result = await visual_question_answering(file_path=image_path, question=question)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(_test())

