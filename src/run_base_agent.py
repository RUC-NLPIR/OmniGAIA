"""
run_base_agent.py — Run a baseline agent (Gemini or Qwen) on the OmniGAIA benchmark.

The script reads a JSON dataset, dispatches each question to the selected agent
(with optional media attachments), evaluates the answers against ground truth,
and writes per-item results plus aggregate metrics.

Configuration file:
    config/config.json
"""
import os
import sys
import json
import asyncio
import logging
import base64
import mimetypes
import httpx
import argparse
import re
import time
import math
import random
import cv2
import numpy as np
import hashlib
import threading
import tempfile
import io
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from config_loader import get_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    import whisper
except ImportError:
    whisper = None

# =============================================================================
# Configuration
# =============================================================================

APP_CONFIG = get_config()

SYSTEM_PROMPT = (
    "You are an omni-modal general AI assistant. Please answer the question "
    "provided to you based on the input image, audio, or video content.\n\n"
    "You should think step by step to answer the question. You may use "
    "available tools to assist with your analysis if needed.\n\n"
    "Please provide your final answer using this format: "
    "<answer>YOUR_ANSWER</answer>."
)

ACTIVE_PERCEPTION_PROMPT_NOTE = (
    '**Note:**\n'
    '- If there are segments in the input image/audio/video that are unclear '
    'to you, you should use the "read_image/read_audio/read_video" tool to '
    "examine them carefully to ensure you have correctly perceived the input "
    "media.\n"
)

FORCED_FINAL_ANSWER_PROMPT = (
    "You have reached the maximum tool-use turns. Stop calling tools and provide your final answer now. "
    "Return only one final answer in the format <answer>YOUR_ANSWER</answer>."
)

# Retry Config
MAX_RETRIES = 5
RETRY_DELAY_BASE = 3

# Evaluation LLM (loaded from config/config.json)
EVAL_ENDPOINTS = [
    {
        "base_url": APP_CONFIG["evaluation"]["base_url"],
        "api_key": APP_CONFIG["evaluation"]["api_key"],
        "model": APP_CONFIG["evaluation"]["model"],
    },
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("BaseAgent")

# Category ordering (short label -> full label in data)
CATEGORY_ORDER = ["Geo.", "Tech.", "Hist.", "Fin.", "Sport", "Art", "Movie", "Sci.", "Food"]
CATEGORY_LABEL_MAP = {
    "Geo.": "Geography & Travel",
    "Tech.": "Technology",
    "Hist.": "History & Society",
    "Fin.": "Finance & Commerce",
    "Sport": "Sports",
    "Art": "Arts & Culture",
    "Movie": "Movies",
    "Sci.": "Science & Nature",
    "Food": "Food & Nutrition",
}

_EVAL_HTTP_CLIENT = None


def _get_eval_http_client() -> httpx.AsyncClient:
    """Lazy-initialise a shared HTTP client for evaluation requests."""
    global _EVAL_HTTP_CLIENT
    if _EVAL_HTTP_CLIENT is None:
        _EVAL_HTTP_CLIENT = httpx.AsyncClient(timeout=3600, trust_env=False)
    return _EVAL_HTTP_CLIENT


def _get_eval_client() -> Tuple[Any, str]:
    """Return an ``AsyncOpenAI`` client + model name for answer evaluation."""
    endpoint = random.choice(EVAL_ENDPOINTS)
    http_client = _get_eval_http_client()
    return (
        AsyncOpenAI(
            api_key=endpoint["api_key"],
            base_url=endpoint["base_url"],
            http_client=http_client,
        ),
        endpoint["model"],
    )

# =============================================================================
# Tool Definitions & Mock Implementations
# =============================================================================

from tools.web_tools import (
    web_search, 
    page_browser, 
    get_openai_function_web_search,
    get_openai_function_page_browser
)
from tools.code_executor import (
    code_executor,
    get_openai_function_code_executor
)


# =============================================================================
# Tool Definitions (Using actual tools)
# =============================================================================

class AgentTools:
    """
    Wrapper for tool implementations.
    """
    @staticmethod
    async def web_search(query: str) -> Dict[str, Any]:
        logger.info(f"[Tool Call] web_search: {query}")
        return await web_search(query)

    @staticmethod
    async def page_browser(urls: Union[str, List[str]]) -> Dict[str, Any]:
        logger.info(f"[Tool Call] page_browser: {urls}")
        # page_browser in web_tools expects a list of urls
        if isinstance(urls, str):
            urls = [urls]
        return await page_browser(urls)

    @staticmethod
    async def code_executor(code: str) -> Dict[str, Any]:
        logger.info(f"[Tool Call] code_executor: \n{code}")
        return await code_executor(code)

    @staticmethod
    async def read_video(video_id: str, t_start: int, t_end: int) -> Dict[str, Any]:
        logger.info(f"[Tool Call] read_video: {video_id} ({t_start}-{t_end}s)")
        return {
            "status": "success",
            "action": "read_video",
            "video_id": video_id,
            "t_start": t_start,
            "t_end": t_end,
            "message": f"Reading video {video_id} from {t_start}s to {t_end}s. Data will follow.",
        }

    @staticmethod
    async def read_audio(audio_id: str, t_start: int, t_end: int) -> Dict[str, Any]:
        logger.info(f"[Tool Call] read_audio: {audio_id} ({t_start}-{t_end}s)")
        return {
            "status": "success",
            "action": "read_audio",
            "audio_id": audio_id,
            "t_start": t_start,
            "t_end": t_end,
            "message": f"Reading audio {audio_id} from {t_start}s to {t_end}s. Data will follow.",
        }

    @staticmethod
    async def read_image(image_ids: List[str], crop_box: Optional[List[int]] = None) -> Dict[str, Any]:
        logger.info(f"[Tool Call] read_image: {image_ids}, crop_box={crop_box}")
        return {
            "status": "success",
            "action": "read_image",
            "image_ids": image_ids,
            "crop_box": crop_box,
            "message": f"Reading images {image_ids}. Data will follow.",
        }
        
# Gemini Tool Schemas - Adapted from OpenAI schemas in tools
def _convert_openai_schema_to_gemini(openai_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAI function schema to Gemini function declaration.
    Gemini uses a slightly different format (camelCase and specific type names).
    """
    func = openai_schema['function']
    
    # Map types to Gemini types (string, integer, number, boolean, array, object)
    # Note: Use lowercase for REST API JSON
    def map_type(t: str) -> str:
        t_map = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object"
        }
        return t_map.get(t.lower(), "string")
        
    parameters = func['parameters']
    gemini_properties = {}
    
    for prop_name, prop_def in parameters['properties'].items():
        gemini_prop = {
            "type": map_type(prop_def['type']),
            "description": prop_def.get('description', '')
        }
        if 'items' in prop_def:
             gemini_prop['items'] = {"type": map_type(prop_def['items']['type'])}
             if 'format' in prop_def['items']:
                 # Gemini doesn't strictly use format, but description helps
                 pass
        
        gemini_properties[prop_name] = gemini_prop
        
    return {
        "name": func['name'],
        "description": func['description'],
        "parameters": {
            "type": "object",
            "properties": gemini_properties,
            "required": parameters.get('required', [])
        }
    }

GEMINI_TOOLS_SCHEMA = [
    {
        "functionDeclarations": [
            _convert_openai_schema_to_gemini(get_openai_function_web_search()),
            _convert_openai_schema_to_gemini(get_openai_function_page_browser()),
            _convert_openai_schema_to_gemini(get_openai_function_code_executor())
        ]
    }
]

OPENAI_TOOLS_SCHEMA = [
    get_openai_function_web_search(),
    get_openai_function_page_browser(),
    get_openai_function_code_executor()
]


def get_openai_function_read_video() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_video",
            "description": "Reads a specific time segment of a video file to examine details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string", "description": "The video identifier or filename."},
                    "t_start": {"type": "integer", "description": "Start time in seconds."},
                    "t_end": {"type": "integer", "description": "End time in seconds."},
                },
                "required": ["video_id", "t_start", "t_end"],
            },
        },
    }


def get_openai_function_read_audio() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_audio",
            "description": "Reads a specific time segment of an audio file to listen to details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_id": {"type": "string", "description": "The audio identifier or filename."},
                    "t_start": {"type": "integer", "description": "Start time in seconds."},
                    "t_end": {"type": "integer", "description": "End time in seconds."},
                },
                "required": ["audio_id", "t_start", "t_end"],
            },
        },
    }


def get_openai_function_read_image() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_image",
            "description": (
                "Reads specific images to view them in detail. Optionally crop the image by "
                "providing a crop box [left, top, right, bottom]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of image identifiers or filenames.",
                    },
                    "crop_box": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Optional. [left, top, right, bottom] crop coordinates.",
                    },
                },
                "required": ["image_ids"],
            },
        },
    }

# =============================================================================
# Agent Implementation
# =============================================================================

class GeminiBaseAgent:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base_url: str,
        request_timeout: int = 600,
        forced_final_timeout: int = 300,
        ffmpeg_timeout: int = 180,
    ):
        self.api_key = api_key
        self.model = model
        self.request_timeout = request_timeout
        self.forced_final_timeout = forced_final_timeout
        self.ffmpeg_timeout = ffmpeg_timeout
        # Disable proxy by setting trust_env=False to ignore HTTP_PROXY/HTTPS_PROXY environment variables
        self.http_client = httpx.AsyncClient(timeout=3600, trust_env=False)
        self.request_url = api_base_url
        self.tools = AgentTools()

    async def _file_to_base64_part(self, path: str, input_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Reads file and converts to Gemini Inline Data format."""
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
            
        try:
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type:
                # Fallbacks
                if path.endswith(".mp4"): mime_type = "video/mp4"
                elif path.endswith(".mp3"): mime_type = "audio/mpeg"
                elif path.endswith(".wav"): mime_type = "audio/wav"
                elif path.endswith(".jpg") or path.endswith(".jpeg"): mime_type = "image/jpeg"
                elif path.endswith(".png"): mime_type = "image/png"
                else: 
                    # Use input_type to provide a default fallback if possible
                    if input_type == "image": mime_type = "image/jpeg"
                    elif input_type == "audio": mime_type = "audio/wav"
                    elif input_type == "video": mime_type = "video/mp4"
                    else: mime_type = "application/octet-stream"

            # Correct mime_type based on input_type if provided
            if input_type:
                if input_type == "image" and not mime_type.startswith("image/"):
                     mime_type = "image/jpeg"
                elif input_type == "audio" and not mime_type.startswith("audio/"):
                     mime_type = "audio/wav"
                elif input_type == "video" and not mime_type.startswith("video/"):
                     mime_type = "video/mp4"

            # Handle audio truncation and transcoding if pydub is available
            if (input_type == "audio" or mime_type.startswith("audio/")) and AudioSegment:
                try:
                    loop = asyncio.get_running_loop()
                    def _process_audio():
                        audio = AudioSegment.from_file(path)
                        # Apply LlamaFactory compression strategy: Mono + 16kHz
                        audio = audio.set_channels(1)
                        audio = audio.set_frame_rate(16000)

                        fifteen_mins_ms = 30 * 60 * 1000
                        if len(audio) > fifteen_mins_ms:
                            logger.info(f"Audio {os.path.basename(path)} > 30m. Truncating to 30m.")
                            audio = audio[:fifteen_mins_ms]
                        
                        buf = io.BytesIO()
                        audio.export(buf, format="mp3")
                        return base64.b64encode(buf.getvalue()).decode("utf-8"), "audio/mpeg"
                    
                    if self.ffmpeg_timeout and self.ffmpeg_timeout > 0:
                        b64_data, mime_type = await asyncio.wait_for(
                            loop.run_in_executor(None, _process_audio),
                            timeout=self.ffmpeg_timeout,
                        )
                    else:
                        b64_data, mime_type = await loop.run_in_executor(None, _process_audio)
                except Exception as e:
                    logger.warning(f"Pydub processing failed for {path}: {e}. Falling back to raw read.")
                    loop = asyncio.get_running_loop()
                    with open(path, "rb") as f:
                        # Fallback to 20MB truncation if pydub fails
                        max_audio_size = 20 * 1024 * 1024
                        data = await loop.run_in_executor(None, f.read, max_audio_size)
                        b64_data = base64.b64encode(data).decode("utf-8")
            else:
                # Blocking read is fine for this test script, but using asyncio to be consistent
                loop = asyncio.get_running_loop()
                with open(path, "rb") as f:
                    data = await loop.run_in_executor(None, f.read)
                    b64_data = base64.b64encode(data).decode("utf-8")
                
            return {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": b64_data
                }
            }
        except Exception as e:
            logger.error(f"Failed to encode file {path}: {e}")
            return None

    def _extract_frames(self, video_path: str, max_frames: int = 30, target_fps: float = 1.0) -> List[Dict[str, Any]]:
        frames = []
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return frames
            
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or video_fps <= 0:
                cap.release()
                logger.warning(f"Invalid video metadata for {video_path}: frames={total_frames}, fps={video_fps}")
                return frames

            duration = total_frames / video_fps
            
            # Smart frame sampling strategy
            nframes = int(duration * target_fps)
            min_frames = 4
            floor_frames = min(min_frames, max_frames)
            
            if total_frames < floor_frames:
                nframes = total_frames
            else:
                nframes = min(nframes, max_frames)
                nframes = max(floor_frames, nframes)
                
            if nframes == total_frames:
                sample_indices = np.arange(total_frames)
            else:
                sample_indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
            
            logger.info(f"Extracting {len(sample_indices)} frames from {video_path} (Duration: {duration:.2f}s, FPS: {video_fps}, Max limit: {max_frames})")

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Smart Resize
                height, width = frame.shape[:2]
                new_height, new_width = smart_resize(height, width)
                
                if new_height != height or new_width != width:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    
                _, buffer = cv2.imencode('.jpg', frame)
                b64_image = base64.b64encode(buffer).decode('utf-8')
                
                frames.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": b64_image
                    }
                })
            cap.release()
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            
        return frames

    def _compress_video(self, video_path: str, max_frames: int = 128) -> Optional[Dict[str, Any]]:
        """
        Compresses a video by extracting frames and re-encoding them into a smaller video file.
        Returns a dict compatible with Gemini Inline Data or None.
        """
        if not os.path.exists(video_path):
            return None
            
        temp_video_path = None
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0 or video_fps <= 0:
                cap.release()
                return None

            # LlamaFactory strategy: 2.0 FPS, Max 128 Frames
            target_fps = 2.0
            duration = total_frames / video_fps
            
            # Calculate indices based on FPS
            nframes = int(duration * target_fps)
            nframes = min(nframes, max_frames)
            # Ensure at least 1 frame if duration > 0
            nframes = max(1, nframes)

            sample_indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
            
            # Read first frame to determine size
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
                
            height, width = frame.shape[:2]
            # LlamaFactory strategy: Max 256x256 for video frames
            new_height, new_width = smart_resize(height, width, factor=16, min_pixels=224*224, max_pixels=512*512)
            
            # Create temp file
            fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            
            # Initialize VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_fps = 2.0 # Match sampling FPS
            out = cv2.VideoWriter(temp_video_path, fourcc, out_fps, (new_width, new_height))
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                if new_height != height or new_width != width:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                out.write(frame)
                
            cap.release()
            out.release()
            
            # Check size
            comp_size = os.path.getsize(temp_video_path) / (1024*1024)
            logger.info(f"Compressed video {os.path.basename(video_path)} to {comp_size:.2f}MB ({len(sample_indices)} frames)")
            
            # Read back and base64 encode
            with open(temp_video_path, "rb") as f:
                video_data = f.read()
                b64_video = base64.b64encode(video_data).decode("utf-8")
                
            return {
                "inlineData": {
                    "mimeType": "video/mp4",
                    "data": b64_video
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to compress video {video_path}: {e}")
            return None
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    pass

    async def _call_gemini_api(
        self,
        contents: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        request_timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Raw API call with retries."""
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            }
        }
        
        if tools:
            payload["tools"] = tools
        
        # Log the request for debugging
        logger.debug(f"Request URL: {self.request_url}")
        logger.debug(f"Payload keys: {payload.keys()}")

        async def _do_request():
            response = await self.http_client.post(
                self.request_url, 
                headers=headers, 
                content=json.dumps(payload),
                timeout=request_timeout or self.request_timeout
            )
            return response

        for attempt in range(MAX_RETRIES):
            try:
                if semaphore:
                    async with semaphore:
                        response = await _do_request()
                else:
                    response = await _do_request()
                
                if response.status_code != 200:
                    logger.error(f"API Error {response.status_code} (Attempt {attempt+1}): {response.text}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                        continue
                    return None
                
                return response.json()
            except Exception as e:
                logger.error(f"Request failed (Attempt {attempt+1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                    continue
                return None
        return None

    async def _force_final_answer(
        self,
        contents: List[Dict[str, Any]],
        all_tool_calls: List[Dict[str, Any]],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        """Force one final answer generation after max turns."""
        contents.append({"role": "user", "parts": [{"text": FORCED_FINAL_ANSWER_PROMPT}]})
        response_json = await self._call_gemini_api(
            contents,
            tools=None,
            semaphore=semaphore,
            request_timeout=self.forced_final_timeout,
        )
        if not response_json:
            return {
                "output": "Max turns reached without final answer.",
                "tool_calls": all_tool_calls,
                "messages": contents,
            }
        try:
            candidate = response_json["candidates"][0]
            model_content = candidate["content"]
            contents.append(model_content)
            parts = model_content.get("parts", [])
            final_ans = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
            final_ans = final_ans or "Max turns reached without final answer."
            return {"output": final_ans, "tool_calls": all_tool_calls, "messages": contents}
        except Exception as e:
            logger.error(f"Failed to parse forced final answer: {e}")
            return {
                "output": "Max turns reached without final answer.",
                "tool_calls": all_tool_calls,
                "messages": contents,
            }

    async def run(self, question: str, media_items: List[Union[str, Dict[str, str]]] = [], semaphore: Optional[asyncio.Semaphore] = None) -> Dict[str, Any]:
        """
        Main Agent Loop:
        1. Construct initial message (Text + Media).
        2. Call Model.
        3. If Tool Call -> Execute -> Append Result -> Loop.
        4. If Text -> Return.
        """
        logger.info(f"Starting Agent with question: {question}")
        
        # 1. Build Initial Content
        media_parts = []
        
        frames_per_video = 128 # LlamaFactory default

        for item in media_items:
            path = item
            input_type = None
            if isinstance(item, dict):
                path = item.get("path")
                input_type = item.get("type")
            
            if path:
                # Check video size for compression
                is_video = False
                if input_type == "video":
                    is_video = True
                elif path.endswith(('.mp4', '.mkv', '.avi', '.webm')):
                    is_video = True
                
                if is_video:
                    # Always apply LlamaFactory compression strategy regardless of size
                    logger.info(f"Processing video {os.path.basename(path)} with LlamaFactory strategy (2FPS, Max 128 frames, 256x256).")
                    
                    # 1. Compress Video
                    compressed_video_part = self._compress_video(path, max_frames=frames_per_video)
                    if compressed_video_part:
                        media_parts.append(compressed_video_part)
                    
                    # 2. Extract Audio (First 15 mins)
                    if AudioSegment:
                        try:
                            loop = asyncio.get_running_loop()
                            def _extract_audio_from_video():
                                try:
                                    audio = AudioSegment.from_file(path)
                                    # Apply LlamaFactory compression strategy: Mono + 16kHz
                                    audio = audio.set_channels(1)
                                    audio = audio.set_frame_rate(16000)

                                    max_mins_ms = 30 * 60 * 1000  # 30 minutes
                                    if len(audio) > max_mins_ms:
                                        audio = audio[:max_mins_ms]
                                    buf = io.BytesIO()
                                    audio.export(buf, format="mp3")
                                    return base64.b64encode(buf.getvalue()).decode("utf-8")
                                except IndexError:
                                    return None
                            
                            if self.ffmpeg_timeout and self.ffmpeg_timeout > 0:
                                b64_audio = await asyncio.wait_for(
                                    loop.run_in_executor(None, _extract_audio_from_video),
                                    timeout=self.ffmpeg_timeout,
                                )
                            else:
                                b64_audio = await loop.run_in_executor(None, _extract_audio_from_video)
                            
                            if b64_audio:
                                media_parts.append({
                                    "inlineData": {
                                        "mimeType": "audio/mpeg",
                                        "data": b64_audio
                                    }
                                })
                                logger.info(f"Extracted audio from {os.path.basename(path)} (Size: {len(b64_audio)/(1024*1024):.2f}MB base64)")
                            else:
                                logger.warning(f"Failed to extract audio from video {path}: No audio stream found.")

                        except Exception as e:
                            logger.warning(f"Failed to extract audio from video {path}: {e}")
                else:
                    # Static image or audio
                    # For images, we should also ensure max pixels logic if not already handling it?
                    # The current _file_to_base64_part reads raw bytes.
                    # To strictly follow "same strategy", we should resize images to 1024x1024 (or 768x768).
                    # But Gemini handles large images well. I will assume "video logic" was the main concern.
                    # However, if audio > 15m, it is already truncated in _file_to_base64_part which aligns with budget.
                    media_part = await self._file_to_base64_part(path, input_type=input_type)
                    if media_part:
                        media_parts.append(media_part)
        
        # Combine question and media
        contents = [
            {"role": "user", "parts": [{"text": question}] + media_parts}
        ]
        
        try:
            # History maintains the conversation state
            
            # Max turns to prevent infinite loops
            max_turns = 50
            current_turn = 0
            
            all_tool_calls = []

            while current_turn < max_turns:
                current_turn += 1
                logger.info(f"Turn {current_turn}...")
                
                # Call API
                response_json = await self._call_gemini_api(contents, tools=GEMINI_TOOLS_SCHEMA, semaphore=semaphore)
                
                if not response_json:
                    return {"output": "Error: API call failed.", "tool_calls": all_tool_calls, "messages": contents}
                
                try:
                    candidate = response_json['candidates'][0]
                    model_content = candidate['content'] # {"role": "model", "parts": [...]}
                    contents.append(model_content) # Add model response to history
                    
                    parts = model_content.get('parts', [])
                    if not parts:
                        return {"output": "Error: Empty response from model.", "tool_calls": all_tool_calls, "messages": contents}

                    # Check for Function Calls
                    function_calls = []
                    text_response = []
                    
                    for part in parts:
                        if 'functionCall' in part:
                            function_calls.append(part['functionCall'])
                            all_tool_calls.append(part['functionCall'])
                        if 'text' in part:
                            text_response.append(part['text'])
                    
                    # If text only, we are done
                    if not function_calls:
                        final_ans = "".join(text_response)
                        logger.info(f"Final Answer: {final_ans}")
                        return {"output": final_ans, "tool_calls": all_tool_calls, "messages": contents}
                    
                    # Handle Function Calls - Execute all tools in parallel
                    async def execute_tool(fc):
                        fn_name = fc['name']
                        fn_args = fc.get('args', {})
                        
                        logger.info(f"Executing tool: {fn_name} with args: {fn_args}")
                        
                        # Execute Tool
                        tool_result = {}
                        max_tool_retries = 3
                        
                        if hasattr(self.tools, fn_name):
                            tool_func = getattr(self.tools, fn_name)
                            for attempt in range(max_tool_retries):
                                try:
                                    # All our actual tools are async now
                                    tool_result = await tool_func(**fn_args)
                                    if not tool_result and attempt < max_tool_retries - 1:
                                        logger.warning(f"Tool {fn_name} returned empty result (Attempt {attempt+1}). Retrying...")
                                        await asyncio.sleep(1)
                                        continue
                                    break
                                except Exception as e:
                                    if attempt < max_tool_retries - 1:
                                        logger.warning(f"Tool {fn_name} failed (Attempt {attempt+1}): {e}. Retrying...")
                                        await asyncio.sleep(1)
                                    else:
                                        tool_result = {"error": str(e)}
                        else:
                            tool_result = {"error": f"Tool {fn_name} not found."}
                        
                        # Format tool result as text (more compatible with different Gemini gateways)
                        result_text = f"[Tool: {fn_name}]\nResult: {json.dumps(tool_result, ensure_ascii=False, indent=2)}"
                        return result_text
                    
                    # Execute all tools in parallel
                    tool_results_text = await asyncio.gather(*[execute_tool(fc) for fc in function_calls])
                    
                    # Append Function Results as a user message (text format for compatibility)
                    # This avoids issues with different functionResponse format implementations
                    combined_results = "\n\n".join(tool_results_text)
                    contents.append({
                        "role": "user", 
                        "parts": [{"text": combined_results}]
                    })
                    
                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing response: {e}")
                    error_msg = "Error parsing model response."
                    contents.append({"role": "model", "parts": [{"text": error_msg}]})
                    return {"output": error_msg, "tool_calls": all_tool_calls, "messages": contents}
            
            return await self._force_final_answer(contents, all_tool_calls, semaphore=semaphore)
        
        except Exception as e:
            logger.error(f"Error in Gemini run loop: {e}")
            error_msg = f"Error: {str(e)}"
            contents.append({"role": "model", "parts": [{"text": error_msg}]})
            return {"output": error_msg, "tool_calls": all_tool_calls, "messages": contents}

    async def close(self):
        await self.http_client.aclose()

class QwenBaseAgent:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base_url: str,
        enable_active_perception: bool = False,
        request_timeout: int = 600,
        forced_final_timeout: int = 300,
        ffmpeg_timeout: int = 180,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url
        self.enable_active_perception = enable_active_perception
        self.request_timeout = request_timeout
        self.forced_final_timeout = forced_final_timeout
        self.ffmpeg_timeout = ffmpeg_timeout
        if AsyncOpenAI is None:
            raise ImportError("openai package is required for QwenBaseAgent")
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
        self.tools = AgentTools()
        self.info_manager = OmniInfoManager()

    def _is_url(self, s: str) -> bool:
        return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("data:"))

    def _to_data_url(self, path: str, default_mime: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or default_mime
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            logger.error(f"Failed to convert {path} to data URL: {e}")
            raise

    def _build_part(self, kind: str, source: str) -> Optional[Dict[str, Any]]:
        default_mimes = {
            "image": "image/jpeg",
            "audio": "audio/wav",
            "video": "video/mp4",
        }
        if self._is_url(source):
            url = source
        else:
            if not os.path.exists(source):
                logger.warning(f"{kind} file not found: {source}")
                return None
            url = self._to_data_url(source, default_mimes[kind])

        return {
            "type": f"{kind}_url",
            f"{kind}_url": {"url": url},
        }

    def _build_tools_prompt(self, tools_schema: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build Hermes-style tools prompt for manual tool calling."""
        tools_json = []
        active_schema = tools_schema if tools_schema is not None else OPENAI_TOOLS_SCHEMA
        for tool in active_schema:
            func = tool["function"]
            tools_json.append(json.dumps(func, ensure_ascii=False))

        tools_str = "\n".join(tools_json)

        return f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "<function-name>", "arguments": <args-json-object>}}
</tool_call>"""

    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Parse <tool_call> tags from model output."""
        tool_calls = []
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, content or "", re.DOTALL)

        for i, match in enumerate(matches):
            try:
                tool_call_data = json.loads(match)
                tool_calls.append(
                    {
                        "id": f"call_{i}_{int(time.time()*1000)}",
                        "name": tool_call_data.get("name"),
                        "arguments": tool_call_data.get("arguments", {}),
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call JSON: {match}, error: {e}")
                continue

        return tool_calls

    def _normalize_media_id(self, media_id: Any) -> str:
        if media_id is None:
            return ""
        mid = str(media_id).strip()
        if mid.startswith("<") and mid.endswith(">") and len(mid) > 2:
            mid = mid[1:-1].strip()
        return mid

    def _parse_time_value(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value.strip()
            if not v or v.lower() in {"none", "null"}:
                return None
            if v.endswith("s"):
                v = v[:-1].strip()
            try:
                return float(v)
            except ValueError:
                return None
        return None

    def _register_media_id(self, media_id: str, path: str, mapping: Dict[str, str]) -> None:
        if not media_id:
            return
        if media_id not in mapping:
            mapping[media_id] = path
        if media_id.startswith(("image_", "audio_", "video_")):
            stripped = media_id.split("_", 1)[1]
            if stripped and stripped not in mapping:
                mapping[stripped] = path

    def _build_media_id_map(self, media_items: List[Union[str, Dict[str, Any]]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for item in media_items:
            if isinstance(item, dict):
                path = item.get("path")
            else:
                path = item
            if not path or self._is_url(path) or not os.path.exists(path):
                continue
            base = os.path.splitext(os.path.basename(path))[0]
            self._register_media_id(base, path, mapping)
            if isinstance(item, dict):
                kind = item.get("type")
                if kind:
                    self._register_media_id(f"{kind}_{base}", path, mapping)
        return mapping

    def _resolve_media_path(self, media_id: str, media_id_to_path: Dict[str, str]) -> Optional[str]:
        normalized = self._normalize_media_id(media_id)
        if not normalized:
            return None
        mapped_path = media_id_to_path.get(normalized)
        if mapped_path and os.path.exists(mapped_path):
            return mapped_path
        return self.info_manager.get_file_path(normalized)

    def _detect_media_types(self, media_items: List[Union[str, Dict[str, Any]]]) -> Dict[str, bool]:
        flags = {"image": False, "audio": False, "video": False}
        for item in media_items:
            if isinstance(item, dict):
                path = item.get("path")
                kind = item.get("type", "")
            else:
                path = item
                kind = ""
            if not path or self._is_url(path):
                continue
            if not kind:
                if path.endswith((".mp4", ".mkv", ".avi", ".webm")):
                    kind = "video"
                elif path.endswith((".wav", ".mp3", ".flac", ".m4a")):
                    kind = "audio"
                else:
                    kind = "image"
            if kind in flags:
                flags[kind] = True
        return flags

    def _build_active_perception_tools_schema(self, media_items: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        tools_schema = list(OPENAI_TOOLS_SCHEMA)
        media_flags = self._detect_media_types(media_items)
        if media_flags["video"]:
            tools_schema.append(get_openai_function_read_video())
        if media_flags["audio"]:
            tools_schema.append(get_openai_function_read_audio())
        if media_flags["image"]:
            tools_schema.append(get_openai_function_read_image())
        return tools_schema

    def _compress_video_segment(
        self,
        video_path: str,
        t_start: float,
        t_end: float,
        max_frames: int = 128,
    ) -> Optional[Dict[str, Any]]:
        if not os.path.exists(video_path):
            return None

        cap = None
        temp_video_path = None
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if total_frames <= 0 or video_fps <= 0:
                return None

            start_frame = int(max(0.0, t_start) * video_fps)
            end_frame = int(max(t_start, t_end) * video_fps)
            end_frame = min(end_frame, total_frames)
            if start_frame >= end_frame:
                return None

            seg_cnt = end_frame - start_frame
            duration = seg_cnt / video_fps
            target_fps = min(2.0, max_frames / duration) if duration > 0 else 2.0
            nframes = int(duration * target_fps)
            nframes = max(1, min(nframes, max_frames))
            sample_indices = np.linspace(start_frame, end_frame - 1, nframes, dtype=int)

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_indices[0]))
            ret, frame = cap.read()
            if not ret:
                return None

            height, width = frame.shape[:2]
            new_height, new_width = smart_resize(
                height, width, factor=28, min_pixels=16 * 16, max_pixels=256 * 256
            )

            fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video_path, fourcc, max(0.5, target_fps), (new_width, new_height))

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                if new_height != height or new_width != width:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                out.write(frame)
            out.release()

            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                return None

            with open(temp_video_path, "rb") as f:
                b64_video = base64.b64encode(f.read()).decode("utf-8")
            return {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64_video}"}}
        except Exception as e:
            logger.error(f"Failed to compress video segment {video_path}: {e}")
            return None
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass

    def _compress_audio_segment(self, audio_path: str, t_start: float, t_end: float) -> Optional[Dict[str, Any]]:
        if AudioSegment is None:
            return None
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            start_ms = int(max(0.0, t_start) * 1000)
            end_ms = int(max(t_start, t_end) * 1000)
            segment = audio[start_ms:end_ms]
            if len(segment) > 300 * 1000:
                segment = segment[: 300 * 1000]

            buf = io.BytesIO()
            segment.export(buf, format="mp3")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return self._build_part("audio", f"data:audio/mpeg;base64,{b64}")
        except Exception as e:
            logger.error(f"Audio compression failed for {audio_path}: {e}")
            return None

    async def _handle_active_perception_tool(
        self,
        fn_name: str,
        fn_args: Dict[str, Any],
        media_id_to_path: Dict[str, str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        media_parts: List[Dict[str, Any]] = []

        if fn_name == "read_video":
            vid = self._normalize_media_id(fn_args.get("video_id"))
            t_start = self._parse_time_value(fn_args.get("t_start")) or 0.0
            t_end = self._parse_time_value(fn_args.get("t_end"))
            path = self._resolve_media_path(vid, media_id_to_path)
            if not path:
                return {"status": "error", "message": f"Video {vid} not found."}, media_parts
            if not path.endswith((".mp4", ".mkv", ".avi", ".webm")):
                return {"status": "error", "message": f"Media {vid} is not a video file."}, media_parts

            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            duration = total_frames / fps if fps > 0 else 0.0
            if duration <= 0:
                return {"status": "error", "message": f"Video {vid} has invalid metadata."}, media_parts

            real_end = t_end if t_end is not None else duration
            real_end = min(duration, max(t_start + 0.5, real_end))
            if real_end <= t_start:
                return {"status": "error", "message": "Invalid video segment range."}, media_parts

            segment_part = self._compress_video_segment(path, t_start=t_start, t_end=real_end)
            if not segment_part:
                return {"status": "error", "message": "Could not extract video segment."}, media_parts

            segment_id = f"{vid}_segment_{t_start:.2f}_{real_end:.2f}"
            media_info = (
                f"Media ID: {segment_id}\n"
                f"Media Type: Video Segment\n"
                f"Segment Duration: {real_end - t_start:.2f}s (from {t_start:.2f}s to {real_end:.2f}s)"
            )
            media_parts.extend([{"type": "text", "text": media_info}, segment_part])
            return {
                "status": "success",
                "message": f"Video segment loaded. Assigned Media ID: {segment_id}",
            }, media_parts

        if fn_name == "read_audio":
            aid = self._normalize_media_id(fn_args.get("audio_id"))
            t_start = self._parse_time_value(fn_args.get("t_start")) or 0.0
            t_end = self._parse_time_value(fn_args.get("t_end"))
            path = self._resolve_media_path(aid, media_id_to_path)
            if not path:
                return {"status": "error", "message": f"Audio {aid} not found."}, media_parts
            if not path.endswith((".wav", ".mp3", ".flac", ".m4a", ".mp4", ".mkv", ".avi", ".webm")):
                return {"status": "error", "message": f"Media {aid} does not support audio reading."}, media_parts

            real_end = t_end
            if AudioSegment is not None:
                try:
                    audio_len_s = len(AudioSegment.from_file(path)) / 1000.0
                    if real_end is None:
                        real_end = audio_len_s
                    real_end = min(audio_len_s, max(t_start + 0.5, real_end))
                except Exception:
                    if real_end is None:
                        real_end = t_start + 30.0
            else:
                if real_end is None:
                    real_end = t_start + 30.0
            if real_end <= t_start:
                return {"status": "error", "message": "Invalid audio segment range."}, media_parts

            segment_part = self._compress_audio_segment(path, t_start=t_start, t_end=real_end)
            if not segment_part:
                return {"status": "error", "message": "Could not extract audio segment."}, media_parts

            segment_id = f"{aid}_segment_{t_start:.2f}_{real_end:.2f}"
            media_info = (
                f"Media ID: {segment_id}\n"
                f"Media Type: Audio Segment\n"
                f"Segment Duration: {real_end - t_start:.2f}s (from {t_start:.2f}s to {real_end:.2f}s)"
            )
            media_parts.extend([{"type": "text", "text": media_info}, segment_part])
            return {
                "status": "success",
                "message": f"Audio segment loaded. Assigned Media ID: {segment_id}",
            }, media_parts

        if fn_name == "read_image":
            image_ids = fn_args.get("image_ids", [])
            if isinstance(image_ids, str):
                image_ids = [image_ids]
            crop_box = fn_args.get("crop_box")
            if not isinstance(crop_box, list) or len(crop_box) != 4:
                crop_box = None

            loaded_ids = []
            for iid in image_ids:
                iid = self._normalize_media_id(iid)
                path = self._resolve_media_path(iid, media_id_to_path)
                if not path:
                    continue
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                detail_id = f"{iid}_detail"
                type_desc = "Image Detail"

                if crop_box:
                    x1, y1, x2, y2 = [int(v) for v in crop_box]
                    x1 = max(0, min(x1, w))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h))
                    y2 = max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        img = img[y1:y2, x1:x2]
                        h, w = img.shape[:2]
                        detail_id = f"{iid}_crop_{x1}_{y1}_{x2}_{y2}"
                        type_desc = f"Image Crop (Box: {x1},{y1},{x2},{y2})"

                ok, buf = cv2.imencode(".jpg", img)
                if not ok:
                    continue
                b64 = base64.b64encode(buf).decode("utf-8")
                image_part = self._build_part("image", f"data:image/jpeg;base64,{b64}")
                if not image_part:
                    continue

                media_info = f"Media ID: {detail_id}\nMedia Type: {type_desc}\nResolution: {w}x{h}"
                media_parts.extend([{"type": "text", "text": media_info}, image_part])
                loaded_ids.append(detail_id)

            if loaded_ids:
                return {
                    "status": "success",
                    "message": f"Images loaded. Assigned Media IDs: {loaded_ids}",
                }, media_parts
            return {"status": "error", "message": "No images found."}, media_parts

        return {"status": "error", "message": f"Unsupported active perception tool: {fn_name}"}, media_parts

    async def run(
        self,
        question: str,
        media_items: List[Union[str, Dict[str, str]]] = [],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        content_parts = []
        media_id_to_path = self._build_media_id_map(media_items)

        for item in media_items:
            path = item
            kind = "image"

            if isinstance(item, dict):
                path = item.get("path")
                t = item.get("type", "image")
                if t == "video":
                    kind = "video"
                elif t == "audio":
                    kind = "audio"
                else:
                    kind = "image"
            else:
                if path.endswith((".mp4", ".mkv", ".avi", ".webm")):
                    kind = "video"
                elif path.endswith((".wav", ".mp3", ".flac", ".m4a")):
                    kind = "audio"

            if path:
                part = self._build_part(kind, path)
                if part:
                    content_parts.append(part)

        if question:
            content_parts.append({"type": "text", "text": question})

        tools_schema = OPENAI_TOOLS_SCHEMA
        system_prompt = SYSTEM_PROMPT
        if self.enable_active_perception:
            tools_schema = self._build_active_perception_tools_schema(media_items)
            system_prompt = f"{SYSTEM_PROMPT}\n\n{ACTIVE_PERCEPTION_PROMPT_NOTE}"

        system_content = system_prompt + "\n\n" + self._build_tools_prompt(tools_schema)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_content}]},
            {"role": "user", "content": content_parts},
        ]

        max_turns = 50
        current_turn = 0
        all_tool_calls = []

        while current_turn < max_turns:
            current_turn += 1
            logger.info(f"Turn {current_turn}...")

            try:
                response = await self._run_with_tools(messages, semaphore, all_tool_calls, media_id_to_path)
                if response is not None:
                    response["messages"] = messages
                    return response
                continue
            except Exception as e:
                logger.error(f"Error in Qwen run loop: {e}")
                return {"output": f"Error: {str(e)}", "tool_calls": all_tool_calls, "messages": messages}

        return await self._force_final_answer(messages, all_tool_calls, semaphore=semaphore)

    async def _force_final_answer(
        self,
        messages: List[Dict[str, Any]],
        all_tool_calls: List[Dict[str, Any]],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        """Force one final answer generation after max turns."""
        forced_user_msg = {"role": "user", "content": [{"type": "text", "text": FORCED_FINAL_ANSWER_PROMPT}]}
        forced_messages = messages + [forced_user_msg]

        async def _do_request():
            return await self.client.chat.completions.create(
                model=self.model,
                messages=forced_messages,
                max_tokens=4096,
                timeout=self.forced_final_timeout,
                temperature=0.2,
                top_p=0.9,
                extra_body={
                    "top_k": 20,
                    "repetition_penalty": 1.05,
                },
            )

        try:
            if semaphore:
                async with semaphore:
                    response = await _do_request()
            else:
                response = await _do_request()
            content = response.choices[0].message.content or "Max turns reached without final answer."
            messages.append(forced_user_msg)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
            return {"output": content, "tool_calls": all_tool_calls, "messages": messages}
        except Exception as e:
            logger.error(f"Forced final answer failed: {e}")
            return {"output": "Max turns reached without final answer.", "tool_calls": all_tool_calls, "messages": messages}

    async def _run_with_tools(
        self,
        messages: List[Dict[str, Any]],
        semaphore: Optional[asyncio.Semaphore],
        all_tool_calls: List[Dict[str, Any]],
        media_id_to_path: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Run with manual Hermes-style tool calling (no tools parameter).
        Returns None to continue the loop, or a result dict to finish.
        """

        async def _do_request():
            return await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=20480,
                timeout=self.request_timeout,
                temperature=0.6,
                top_p=0.95,
                extra_body={
                    "top_k": 20,
                    "repetition_penalty": 1.05,
                },
            )

        if semaphore:
            async with semaphore:
                response = await _do_request()
        else:
            response = await _do_request()

        usage_info = None
        if hasattr(response, "usage") and response.usage:
            usage_info = {
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            logger.info(f"Token Usage: {usage_info}")

        content = response.choices[0].message.content or ""
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
        if reasoning_content:
            content = f"<think>{reasoning_content}</think>\n" + content

        cleaned_content = content.strip()
        tool_calls = self._parse_tool_calls(content)

        if tool_calls:
            logger.info(f"Parsed {len(tool_calls)} tool calls from response")
            asst_msg = {"role": "assistant", "content": [{"type": "text", "text": content}]}
            if usage_info:
                asst_msg["usage"] = usage_info
            messages.append(asst_msg)

            async def execute_tool(tc: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
                fn_name = tc.get("name")
                fn_args = tc.get("arguments", {})
                logger.info(f"Executing tool: {fn_name} with args: {fn_args}")

                args_to_pass = fn_args
                if isinstance(args_to_pass, str):
                    try:
                        args_to_pass = json.loads(args_to_pass)
                    except Exception:
                        args_to_pass = {}
                if not isinstance(args_to_pass, dict):
                    args_to_pass = {}

                if not fn_name:
                    return {"name": "unknown", "content": json.dumps({"error": "Missing tool name"}, ensure_ascii=False)}, []

                if self.enable_active_perception and fn_name in {"read_video", "read_audio", "read_image"}:
                    tool_result, media_parts = await self._handle_active_perception_tool(
                        fn_name, args_to_pass, media_id_to_path
                    )
                    return {"name": fn_name, "content": json.dumps(tool_result, ensure_ascii=False)}, media_parts

                tool_result = {}
                max_tool_retries = 3
                if hasattr(self.tools, fn_name):
                    tool_func = getattr(self.tools, fn_name)
                    for attempt in range(max_tool_retries):
                        try:
                            tool_result = await tool_func(**args_to_pass)
                            if not tool_result and attempt < max_tool_retries - 1:
                                logger.warning(
                                    f"Tool {fn_name} returned empty result (Attempt {attempt+1}). Retrying..."
                                )
                                await asyncio.sleep(1)
                                continue
                            break
                        except Exception as e:
                            if attempt < max_tool_retries - 1:
                                logger.warning(f"Tool {fn_name} failed (Attempt {attempt+1}): {e}. Retrying...")
                                await asyncio.sleep(1)
                            else:
                                tool_result = {"error": str(e)}
                else:
                    tool_result = {"error": f"Tool {fn_name} not found."}

                return {"name": fn_name, "content": json.dumps(tool_result, ensure_ascii=False)}, []

            tool_exec_results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])

            for tc in tool_calls:
                all_tool_calls.append(
                    {
                        "id": tc["id"],
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
                        },
                    }
                )

            tool_content_parts: List[Dict[str, Any]] = []
            for tool_result, media_parts in tool_exec_results:
                tool_content_parts.append({"type": "text", "text": tool_result["content"]})
                if media_parts:
                    tool_content_parts.extend(media_parts)
            if not tool_content_parts:
                tool_content_parts = [{"type": "text", "text": "{}"}]

            messages.append({"role": "tool", "content": tool_content_parts})
            return None

        if content:
            log_content = re.sub(r"<think>.*?</think>", "", cleaned_content, flags=re.DOTALL).strip()
            logger.info(f"Final Answer: {log_content}")
            asst_msg = {"role": "assistant", "content": [{"type": "text", "text": content}]}
            if usage_info:
                asst_msg["usage"] = usage_info
            messages.append(asst_msg)
            return {"output": content, "tool_calls": all_tool_calls}
        return {"output": "Error: Empty response from model.", "tool_calls": all_tool_calls}

    async def close(self):
        await self.client.close()

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 1024 * 1024) -> Tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        logger.warning(f"Absolute aspect ratio > 200, skipping resize for safety.")
        return height, width

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar

# =============================================================================
# Helper: ASR Manager
# =============================================================================

class ASRManager:
    def __init__(self, cache_dir="./cache/audio_asr"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model = None
        self.lock = threading.Lock()

    def load_model(self):
        if self.model is None:
             if whisper is None:
                 raise ImportError("whisper not installed")
             logger.info("Loading Whisper model (medium)...")
             self.model = whisper.load_model("medium")

    def get_cache_path(self, file_path):
        try:
            file_stat = os.stat(file_path)
            # Use size and mtime for cache invalidation
            key = f"{os.path.abspath(file_path)}_{file_stat.st_size}_{file_stat.st_mtime}"
            hash_key = hashlib.md5(key.encode("utf-8")).hexdigest()
            return os.path.join(self.cache_dir, f"{hash_key}.json")
        except Exception:
            # Fallback if stat fails
            key = f"{file_path}"
            hash_key = hashlib.md5(key.encode("utf-8")).hexdigest()
            return os.path.join(self.cache_dir, f"{hash_key}.json")

    def transcribe(self, file_path):
        cache_path = self.get_cache_path(file_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"ASR Cache hit for {os.path.basename(file_path)}")
                return data.get("text", "")
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_path}: {e}")

        # Ensure model loading and inference are thread-safe
        with self.lock:
            self.load_model()
            logger.info(f"Transcribing {os.path.basename(file_path)}...")
            try:
                result = self.model.transcribe(file_path)
                text = result.get('text', '').strip()
                
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({"text": text, "path": file_path}, f, ensure_ascii=False)
                
                return text
            except Exception as e:
                logger.error(f"ASR failed for {file_path}: {e}")
                return f"[ASR Error: {str(e)}]"

# =============================================================================
# Helper: OmniInfoManager (Reuse from check_qa_quality_gemini.py logic)
# =============================================================================

class OmniInfoManager:
    """Resolve file paths based on media IDs embedded in questions.

    Configure the data root and media directories in ``config/config.json``.
    The default expected layout is::

        data/
          videos/      # .mp4, .mkv, …
          audios/      # .wav, .mp3, …
          images/      # .jpg, .png, …

    Legacy folder names (video/audio/image) are still supported for backward
    compatibility.
    """

    _DATA_ROOT = APP_CONFIG["paths"]["data_root"]
    _RESOLVED_MEDIA_DIRS = APP_CONFIG.get("data", {}).get("resolved_media_dirs", {})

    BASE_DIRS = {
        "video": _RESOLVED_MEDIA_DIRS.get("video") or [
            os.path.join(_DATA_ROOT, "videos"),
            os.path.join(_DATA_ROOT, "video"),
        ],
        "audio": _RESOLVED_MEDIA_DIRS.get("audio") or [
            os.path.join(_DATA_ROOT, "audios"),
            os.path.join(_DATA_ROOT, "audio"),
        ],
        "image": _RESOLVED_MEDIA_DIRS.get("image") or [
            os.path.join(_DATA_ROOT, "images"),
            os.path.join(_DATA_ROOT, "image"),
        ],
    }
    
    EXTENSIONS = {
        'video': ['.mp4', '.mkv', '.avi', '.webm'],
        'audio': ['.wav', '.mp3', '.flac', '.m4a'],
        'image': ['.jpg', '.jpeg', '.png']
    }

    def __init__(self):
        pass

    def _find_file(self, base_dirs: Union[str, List[str]], raw_id: str, extensions: List[str]) -> Optional[str]:
        if isinstance(base_dirs, str):
            base_dirs = [base_dirs]
            
        for base_dir in base_dirs:
            # Try exact match first if raw_id has extension
            if os.path.exists(os.path.join(base_dir, raw_id)):
                return os.path.join(base_dir, raw_id)
            # Try adding extensions
            for ext in extensions:
                if os.path.exists(os.path.join(base_dir, raw_id + ext)):
                    return os.path.join(base_dir, raw_id + ext)
        return None

    def _clean_id(self, media_id: str) -> str:
        # Remove angle brackets
        s = re.sub(r'[<>]', '', media_id)
        
        # Recursively strip common prefixes
        prefixes = ["video_", "audio_", "image_"]
        for p in prefixes:
            if s.startswith(p):
                s = s[len(p):]
                break
        return s

    def get_file_path(self, media_id: str) -> Optional[str]:
        raw_id = self._clean_id(media_id)
        if not raw_id:
            return None

        if "image" in media_id:
            found = self._find_file(self.BASE_DIRS["image"], raw_id, self.EXTENSIONS["image"])
            if found:
                return found

        if "audio" in media_id:
            found = self._find_file(self.BASE_DIRS["audio"], raw_id, self.EXTENSIONS["audio"])
            if found:
                return found

        if "video" in media_id:
            found = self._find_file(self.BASE_DIRS["video"], raw_id, self.EXTENSIONS["video"])
            if found:
                return found

        return None

    def get_file_paths_for_item(self, item: Dict[str, Any]) -> List[str]:
        """Extract file paths from a QA item's sources list or question string."""
        paths = []

        # Only match specific multimodal ID formats: <image_...>, <audio_...>, <video_...>
        ids = re.findall(r'<((?:image|audio|video)_[^>]+)>', item["question"])
        for mid in ids:
            path = self.get_file_path(mid)
            if path:
                paths.append(path)
            else:
                logger.warning(f"File path not found for Media ID in question: {mid}")
                    
        return list(set(paths)) # Unique paths

    def get_audio_for_video(self, video_path: str) -> Optional[str]:
        """Try to find a corresponding audio file for a given video path."""
        basename = os.path.splitext(os.path.basename(video_path))[0]
        found = self._find_file(self.BASE_DIRS["audio"], basename, self.EXTENSIONS["audio"])
        if found:
            return found
        return None

# =============================================================================
# Helper: Check Equivalence
# =============================================================================

async def check_equivalence(question: str, predicted: str, standard: str, eval_timeout: int = 120) -> Tuple[bool, str]:
    """
    Checks if predicted answer is equivalent to standard answer using an LLM.
    Returns (is_equivalent, full_response_content).
    """
    if not predicted: 
        return False, "No prediction provided."
        
    prompt = f"""Please determine if the model correctly predicted the answer.
Question: {question}
Model Predicted Answer: {predicted}
Labeled Answer: {standard}
Return 'Correct' if the model's prediction is completely accurate, otherwise return 'Incorrect'. Provide only this single word response."""

    if AsyncOpenAI is None:
        raise ImportError("openai package is required for AsyncOpenAI client")
    client, model_name = _get_eval_client()
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=40960,
                extra_body={"chat_template_kwargs": {"thinking": False}},
                timeout=eval_timeout,
            )
            full_content = response.choices[0].message.content or ""
            content = full_content
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            if "Incorrect" in content:
                return False, full_content
            return True, full_content
        except Exception as e:
            logger.error(f"Eval Request Failed (Attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))
                continue
    return False, f"Eval Request Failed after {MAX_RETRIES} attempts."

# =============================================================================
# Helper: Metrics
# =============================================================================

def get_modality_category(item: Dict[str, Any]) -> str:
    """Classifies an item as 'video' or 'audio_image' based on omni_modal_input."""
    omni_input = item.get("omni_modal_input", [])
    if not isinstance(omni_input, list):
        return "audio_image"
    if any(isinstance(inp, dict) and inp.get("type") == "video" for inp in omni_input):
        return "video"
    return "audio_image"

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "count": 0,
            "em": 0.0,
            "llm_equal": 0.0,
            "avg_tool_calls": 0.0,
            "non_empty_ratio": 0.0
        }
    
    count = len(results)
    
    # EM Score
    total_em = sum(r.get("em_score", 0) for r in results)
    avg_em = total_em / count
    
    # LLM Equal
    def is_correct(val):
        if isinstance(val, bool): return val
        if isinstance(val, (int, float)): return val > 0
        return False
        
    total_llm = sum(1 for r in results if is_correct(r.get("llm_equal")))
    avg_llm = total_llm / count
    
    # Tool Calls
    total_tool_calls = sum(r.get("tool_call_num", 0) for r in results)
    avg_tool_calls = total_tool_calls / count
    
    # Non-empty answer
    non_empty = sum(1 for r in results if r.get("predicted_answer"))
    non_empty_ratio = non_empty / count
    
    return {
        "count": count,
        "em": avg_em,
        "llm_equal": avg_llm,
        "avg_tool_calls": avg_tool_calls,
        "non_empty_ratio": non_empty_ratio
    }


def calculate_category_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    category_metrics: Dict[str, Any] = {}
    for short_label in CATEGORY_ORDER:
        full_label = CATEGORY_LABEL_MAP.get(short_label, short_label)
        cat_items = [r for r in results if r.get("category") == full_label]
        category_metrics[short_label] = calculate_metrics(cat_items)
    return category_metrics

# =============================================================================
# Test Execution
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Run baseline agent on QA items.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to the input JSON file containing QA items.")
    parser.add_argument("--max_items", type=int, default=None, help="Limit the number of items to process.")
    parser.add_argument('--concurrent_limit', type=int, default=None, help="Maximum number of concurrent API calls")
    parser.add_argument('--api_key', type=str, default=None, help="API Key")
    parser.add_argument('--api_base_url', type=str, default=None, help="API Base URL")
    parser.add_argument('--level', type=str, default=None, help="Filter items by difficulty level (Easy, Medium, Hard)")
    parser.add_argument('--model_name', type=str, default=None, help="Model Name")
    parser.add_argument('--use_asr', action='store_true', default=False, help="Use ASR to convert audio/video audio to text input")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for results (default: ./outputs)")
    parser.add_argument("--request_timeout", type=int, default=None, help="Per-request timeout in seconds (default: 600)")
    parser.add_argument("--forced_final_timeout", type=int, default=None, help="Timeout for forced final answer after max turns (default: 300)")
    parser.add_argument("--ffmpeg_timeout", type=int, default=None, help="Timeout for ffmpeg-related media processing in seconds (default: 180)")
    parser.add_argument("--item_timeout", type=int, default=None, help="Max total processing time per item in seconds (default: 1800)")
    parser.add_argument("--eval_timeout", type=int, default=None, help="Timeout for LLM equivalence evaluation in seconds (default: 120)")
    parser.add_argument("--skip_eval", action="store_true", default=False, help="Skip LLM-based equivalence evaluation")
    parser.add_argument(
        "--enable-active-perception",
        action="store_true",
        help="Enable read_video/read_audio/read_image active-perception tools for Qwen models.",
    )

    args = parser.parse_args()

    # Apply config file values as fallback for any args not supplied on the CLI.
    # CLI-provided values always take priority; config values fill in the rest.
    _agent_cfg = APP_CONFIG.get("agent", {})

    def _resolve(cli_val, cfg_key, default=None):
        """Return cli_val if explicitly set, else config value, else hard default."""
        if cli_val is not None:
            return cli_val
        cfg_val = _agent_cfg.get(cfg_key)
        if cfg_val is not None:
            return cfg_val
        return default

    args.input_file          = _resolve(args.input_file,          "input_file")
    args.api_key             = _resolve(args.api_key,             "api_key")
    args.api_base_url        = _resolve(args.api_base_url,        "api_base_url")
    args.model_name          = _resolve(args.model_name,          "model_name")
    args.level               = _resolve(args.level,               "level")
    args.max_items           = _resolve(args.max_items,           "max_items")
    args.concurrent_limit    = _resolve(args.concurrent_limit,    "concurrent_limit",    5)
    args.output_dir          = _resolve(args.output_dir,          "output_dir",          "./outputs")
    args.request_timeout     = _resolve(args.request_timeout,     "request_timeout",     600)
    args.forced_final_timeout = _resolve(args.forced_final_timeout, "forced_final_timeout", 300)
    args.ffmpeg_timeout      = _resolve(args.ffmpeg_timeout,      "ffmpeg_timeout",      180)
    args.item_timeout        = _resolve(args.item_timeout,        "item_timeout",        1800)
    args.eval_timeout        = _resolve(args.eval_timeout,        "eval_timeout",        120)
    # Boolean flags: True on CLI always wins; otherwise fall back to config value.
    args.use_asr  = args.use_asr  or bool(_agent_cfg.get("use_asr",  False))
    args.skip_eval = args.skip_eval or bool(_agent_cfg.get("skip_eval", False))

    data_root = APP_CONFIG.get("paths", {}).get("data_root")

    def _resolve_media_path(path_value: Optional[str], input_file_path: Optional[str]) -> Optional[str]:
        """Resolve relative media paths against cwd/data_root/input_file dir.

        For omni_modal_input, metadata often stores paths like "videos/xxx.mp4".
        We keep backward compatibility by trying multiple candidates, but prefer
        existing files first.
        """
        if not isinstance(path_value, str):
            return path_value

        source = path_value.strip()
        if not source:
            return source
        if source.startswith(("http://", "https://", "data:")):
            return source
        if os.path.isabs(source):
            return source

        candidates: List[str] = [source]

        if isinstance(data_root, str) and data_root.strip():
            candidates.append(os.path.join(data_root, source))

        if input_file_path:
            input_dir = os.path.dirname(os.path.abspath(input_file_path))
            candidates.append(os.path.join(input_dir, source))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        if isinstance(data_root, str) and data_root.strip():
            return os.path.join(data_root, source)

        if input_file_path:
            input_dir = os.path.dirname(os.path.abspath(input_file_path))
            return os.path.join(input_dir, source)

        return source
    
    if not args.input_file:  
        print("Please provide an input file using --input_file")
        # Fallback for testing without args
        # args.input_file = "path/to/your/test.json" 
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)
        
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except Exception as e:
        print(f"Failed to load input JSON: {e}")
        sys.exit(1)
        
    if not isinstance(items, list):
        items = [items]
        
    if args.level:
        target_level = args.level.strip().capitalize()
        original_count = len(items)
        items = [item for item in items if item.get("Level") == target_level]
        print(f"Filtered items by level '{target_level}': {len(items)}/{original_count} kept.")

    if args.max_items:
        items = items[:args.max_items]

    model_name_lower = (args.model_name or "").lower()
    is_qwen_like_model = ("qwen" in model_name_lower) or ("omniatlas" in model_name_lower)

    if args.enable_active_perception and not is_qwen_like_model:
        logger.warning("--enable-active-perception is currently only supported in Qwen/OmniAtlas models. Ignoring for this model.")

    semaphore = asyncio.Semaphore(args.concurrent_limit)
    
    # Create agents pool
    agents = []
    
    api_keys = [args.api_key]
    
    for key in api_keys:
        if is_qwen_like_model:
            agent = QwenBaseAgent(
                api_key=key, 
                model=args.model_name, 
                api_base_url=args.api_base_url,
                enable_active_perception=args.enable_active_perception,
                request_timeout=args.request_timeout,
                forced_final_timeout=args.forced_final_timeout,
                ffmpeg_timeout=args.ffmpeg_timeout,
            )
        else:
            agent = GeminiBaseAgent(
                api_key=key, 
                model=args.model_name, 
                api_base_url=args.api_base_url,
                request_timeout=args.request_timeout,
                forced_final_timeout=args.forced_final_timeout,
                ffmpeg_timeout=args.ffmpeg_timeout,
            )
        agents.append(agent)
    
    omni_manager = OmniInfoManager()
    
    asr_manager = None
    if args.use_asr:
        asr_manager = ASRManager()
        # Preload model to ensure thread safety and avoid latency during processing
        try:
            asr_manager.load_model()
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            sys.exit(1)

    # Pre-process all audio with ASR if enabled
    if asr_manager:
        logger.info("Pre-processing audio files with ASR...")
        all_audio_paths = set()
        for item in items:
            # 1. Check omni_modal_input
            if "omni_modal_input" in item and item["omni_modal_input"]:
                for inp in item["omni_modal_input"]:
                    path = inp.get("path") if isinstance(inp, dict) else inp
                    if path:
                        resolved_path = _resolve_media_path(path, args.input_file)
                        if resolved_path:
                            all_audio_paths.add(resolved_path)
            else:
                # Fallback to OmniInfoManager
                paths = omni_manager.get_file_paths_for_item(item)
                for p in paths:
                    all_audio_paths.add(p)
        
        # Filter for audio/video files only
        valid_paths = []
        for p in all_audio_paths:
            if not os.path.exists(p): continue
            if p.endswith(('.wav', '.mp3', '.flac', '.m4a', '.mp4', '.mkv', '.avi', '.webm')):
                valid_paths.append(p)
        
        # Process sequentially with progress bar
        if tqdm:
            for p in tqdm(valid_paths, desc="ASR Pre-processing"):
                asr_manager.transcribe(p)
        else:
            print(f"Processing {len(valid_paths)} files for ASR...")
            for i, p in enumerate(valid_paths):
                if i % 10 == 0: print(f"ASR: {i}/{len(valid_paths)}")
                asr_manager.transcribe(p)
        
        logger.info("ASR Pre-processing complete.")

    async def process_item(i, item, semaphore, agent):
        question = item.get("question", "")
        if not question:
            return None
            
        # Get Media Files
        media_items = []
        
        # 1. Check for omni_modal_input (New Format)
        if "omni_modal_input" in item and item["omni_modal_input"]:
             # Use the provided list of dicts directly
             for inp in item["omni_modal_input"]:
                 if "path" in inp:
                    normalized_inp = dict(inp)
                    normalized_inp["path"] = _resolve_media_path(inp.get("path"), args.input_file)
                    media_items.append(normalized_inp)
        else:
            # Fallback to old logic (OmniInfoManager)
            paths = omni_manager.get_file_paths_for_item(item)
            media_items.extend(paths)
            
        # 2. Check for file_input and append to question
        if "file_input" in item:
             file_paths = []
             for f in item["file_input"]:
                 if "path" in f:
                     file_paths.append(f["path"])
             
             if file_paths:
                 # Append to question
                 question += f"\n\nReferenced Files:\n" + "\n".join(file_paths)
        
        # Handle ASR if enabled
        if asr_manager:
            new_media_items = []
            transcripts = []
            loop = asyncio.get_running_loop()
            
            for itm in media_items:
                path = itm if isinstance(itm, str) else itm.get("path")
                if not path or not os.path.exists(path):
                     new_media_items.append(itm)
                     continue
                
                # Determine type
                is_video = False
                is_audio = False
                if isinstance(itm, dict):
                     t = itm.get("type", "")
                     if t == "video": is_video = True
                     elif t == "audio": is_audio = True
                
                if not is_video and not is_audio:
                     if path.endswith(('.mp4', '.mkv', '.avi', '.webm')):
                          is_video = True
                     elif path.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                          is_audio = True
                
                if is_audio:
                     # Transcribe and replace audio item
                     txt = await loop.run_in_executor(None, asr_manager.transcribe, path)
                     if txt:
                          transcripts.append(f"[Audio Transcript - {os.path.basename(path)}]:\n{txt}")
                elif is_video:
                     # Transcribe audio from video, keep video item for visuals
                     txt = await loop.run_in_executor(None, asr_manager.transcribe, path)
                     if txt:
                          transcripts.append(f"[Video Audio Transcript - {os.path.basename(path)}]:\n{txt}")
                     new_media_items.append(itm)
                else:
                     new_media_items.append(itm)
            
            media_items = new_media_items
            if transcripts:
                 question += "\n\n" + "\n\n".join(transcripts)

        # Special handling for Qwen: add audio track for videos if available
        # This mimics the behavior in extract_omni_info.py where audio is explicitly passed
        if isinstance(agent, QwenBaseAgent) and not args.use_asr:
            additional_audio = []
            processed_videos = set()
            
            existing_paths = set()
            for itm in media_items:
                p = itm if isinstance(itm, str) else itm.get("path")
                if p: existing_paths.add(p)

            for itm in media_items:
                path = itm if isinstance(itm, str) else itm.get("path")
                if not path: continue
                
                is_video = False
                if isinstance(itm, dict) and itm.get("type") == "video":
                    is_video = True
                elif path.endswith(('.mp4', '.mkv', '.avi', '.webm')):
                    is_video = True
                
                if is_video:
                    if path in processed_videos:
                        continue
                    processed_videos.add(path)
                    audio_path = omni_manager.get_audio_for_video(path)
                    if audio_path and audio_path not in existing_paths and audio_path not in additional_audio:
                         additional_audio.append(audio_path)
                    elif AudioSegment:
                        # Fallback: extract audio directly from video and pass as data URL
                        loop = asyncio.get_running_loop()

                        def _extract_audio_from_video(video_path: str) -> Optional[str]:
                            try:
                                audio = AudioSegment.from_file(video_path)
                                audio = audio.set_channels(1)
                                audio = audio.set_frame_rate(16000)
                                max_mins_ms = 30 * 60 * 1000  # 30 minutes
                                if len(audio) > max_mins_ms:
                                    audio = audio[:max_mins_ms]
                                buf = io.BytesIO()
                                audio.export(buf, format="mp3")
                                return base64.b64encode(buf.getvalue()).decode("utf-8")
                            except Exception as e:
                                logger.warning(f"Failed to extract audio from video {video_path}: {e}")
                                return None

                        if args.ffmpeg_timeout and args.ffmpeg_timeout > 0:
                            b64_audio = await asyncio.wait_for(
                                loop.run_in_executor(None, _extract_audio_from_video, path),
                                timeout=args.ffmpeg_timeout,
                            )
                        else:
                            b64_audio = await loop.run_in_executor(None, _extract_audio_from_video, path)
                        if b64_audio:
                            data_url = f"data:audio/mpeg;base64,{b64_audio}"
                            additional_audio.append({"type": "audio", "path": data_url})
                            logger.info(f"Extracted audio from {os.path.basename(path)} for Qwen (fallback data URL).")
            
            if additional_audio:
                # Add audio files to the list
                media_items.extend(additional_audio)
        
        # Run Agent with semaphore
        result_dict = await agent.run(question, media_items=media_items, semaphore=semaphore)
        agent_output = result_dict["output"]
        tool_calls = result_dict["tool_calls"]
        messages = result_dict.get("messages", [])
        
        # Clean up messages (remove large base64 data)
        def _truncate_large_data(obj):
            """Recursively truncate large base64 data in messages."""
            if isinstance(obj, dict):
                new_obj = {}
                for k, v in obj.items():
                    if k == "thoughtSignature":
                        continue
                    # Gemini inlineData
                    if k == "data" and isinstance(v, str) and len(v) > 1000:
                        new_obj[k] = "<base64_data_truncated>"
                    # Qwen data url
                    elif k == "url" and isinstance(v, str) and v.startswith("data:") and len(v) > 1000:
                        new_obj[k] = "<data_url_truncated>"
                    else:
                        new_obj[k] = _truncate_large_data(v)
                return new_obj
            elif isinstance(obj, list):
                return [_truncate_large_data(item) for item in obj]
            else:
                return obj
        
        messages = _truncate_large_data(messages)

        # Extract predicted answer
        matches = re.findall(r'<answer>(.*?)</answer>', agent_output, re.DOTALL)
        if matches:
            predicted_answer = matches[-1].strip()
        else:
            # Fallback: Last 20 words
            words = agent_output.split()
            predicted_answer = " ".join(words[-20:])

        # Prepare text for equivalence check
        eval_predicted_text = predicted_answer
        if not eval_predicted_text:
            # Use response without reasoning
            cleaned_output = re.sub(r'<think>.*?</think>', '', agent_output, flags=re.DOTALL).strip()
            eval_predicted_text = cleaned_output

        # Normalize and calculate EM
        def normalize_text(s):
            if not s: return ""
            s = str(s).lower()
            s = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            return " ".join(s.split())
            
        ground_truth = item.get("answer", "")
        em_score = 1 if normalize_text(predicted_answer) == normalize_text(ground_truth) else 0
        
        # Calculate LLM Equivalence
        if args.skip_eval:
            llm_equal = em_score == 1
            llm_eval_response = "LLM evaluation skipped by --skip_eval."
        elif predicted_answer:
            if em_score == 1:
                llm_equal = True
                llm_eval_response = "EM is 1, skipping LLM evaluation."
            else:
                llm_equal, llm_eval_response = await check_equivalence(
                    question,
                    eval_predicted_text,
                    ground_truth,
                    eval_timeout=args.eval_timeout,
                )
        else:
            llm_equal = False
            llm_eval_response = "Predicted answer is empty, skipping LLM evaluation."
        llm_score = 1 if llm_equal else 0

        # Calculate tool call number
        tool_call_num = len(tool_calls)
        
        # Collect results
        result_item = {
            "question": question,
            "omni_modal_input": item.get("omni_modal_input"),
            "annotated_solution": item.get("annotated_solution"),
            "sources": item.get("sources"),
            "question_type": item.get("question_type"),
            "category": item.get("category"),
            "answer": ground_truth,
            "total_steps": item.get("total_steps"),
            "difficulty": item.get("difficulty"),
            "Level": item.get("Level"),
            "required_external_tools": item.get("required_external_tools"),
            "answerable_without_visual_content": item.get("answerable_without_visual_content"),
            "answerable_without_audio_content": item.get("answerable_without_audio_content"),
            "answerable_without_visual_and_audio_content": item.get("answerable_without_visual_and_audio_content"),
            "answerable_without_tools": item.get("answerable_without_tools"),
            "tool_call_num": tool_call_num,
            "predicted_answer": predicted_answer,
            "em_score": em_score,
            "llm_equal": llm_score,
            "llm_eval_response": llm_eval_response,
            "messages": messages,
        }
        
        return result_item
    
    try:
        print(f"Processing {len(items)} items with concurrency limit {args.concurrent_limit}...")

        async def process_item_with_timeout(i, item, semaphore, agent):
            if args.item_timeout and args.item_timeout > 0:
                try:
                    return await asyncio.wait_for(
                        process_item(i, item, semaphore, agent),
                        timeout=args.item_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Item {i} timed out after {args.item_timeout}s.")
                    return {
                        "question": item.get("question", ""),
                        "omni_modal_input": item.get("omni_modal_input"),
                        "annotated_solution": item.get("annotated_solution"),
                        "sources": item.get("sources"),
                        "question_type": item.get("question_type"),
                        "category": item.get("category"),
                        "answer": item.get("answer", ""),
                        "total_steps": item.get("total_steps"),
                        "difficulty": item.get("difficulty"),
                        "Level": item.get("Level"),
                        "required_external_tools": item.get("required_external_tools"),
                        "answerable_without_visual_content": item.get("answerable_without_visual_content"),
                        "answerable_without_audio_content": item.get("answerable_without_audio_content"),
                        "answerable_without_visual_and_audio_content": item.get("answerable_without_visual_and_audio_content"),
                        "answerable_without_tools": item.get("answerable_without_tools"),
                        "tool_call_num": 0,
                        "predicted_answer": "",
                        "em_score": 0,
                        "llm_equal": 0,
                        "llm_eval_response": f"Item timeout after {args.item_timeout} seconds.",
                        "messages": [],
                    }
            return await process_item(i, item, semaphore, agent)

        tasks = [
            process_item_with_timeout(i, item, semaphore, agents[i % len(agents)])
            for i, item in enumerate(items)
        ]
        
        if tqdm:
            pbar = tqdm(total=len(tasks))
            async def wrap_task(task):
                res = await task
                pbar.update(1)
                return res
            
            wrapped_tasks = [wrap_task(task) for task in tasks]
            results = await asyncio.gather(*wrapped_tasks)
            pbar.close()
        else:
            results = await asyncio.gather(*tasks)

        results = [r for r in results if r is not None]
        
        # Calculate metrics
        overall_metrics = calculate_metrics(results)
        
        # Calculate metrics by Level
        level_metrics = {}
        for level in ["Easy", "Medium", "Hard"]:
            level_results = [r for r in results if r.get("Level") == level]
            level_metrics[level] = calculate_metrics(level_results)
            
        # Calculate metrics by question category in fixed order
        category_metrics = calculate_category_metrics(results)

        # Construct output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = agents[0].model
        output_dir = os.path.join(args.output_dir, f"base_agent_{model_name_safe}")
        os.makedirs(output_dir, exist_ok=True)
        
        avg_em = overall_metrics['em']
        avg_llm_equal = overall_metrics['llm_equal']
        
        output_filename = f"run_{timestamp}_em{avg_em:.4f}_llmeq{avg_llm_equal:.4f}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # Save metrics file
        metrics_data = {
            "overall": overall_metrics,
            "by_level": level_metrics,
            "by_category": category_metrics,
        }
        metrics_filename = f"{os.path.splitext(output_filename)[0]}_metrics.json"
        metrics_path = os.path.join(output_dir, metrics_filename)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        # Print summary statistics
        print(f"\n{'='*50}")
        print(f"Results saved to {output_path}")
        print(f"Metrics saved to {metrics_path}")
        print(f"{'='*50}")
        print(f"Total Items:            {overall_metrics['count']}")
        print(f"Average EM Score:       {overall_metrics['em']:.4f}")
        print(f"Average LLM Equal Score:{overall_metrics['llm_equal']:.4f}")
        print(f"Average Tool Calls:     {overall_metrics['avg_tool_calls']:.2f}")
        print(f"Non-Empty Answer Ratio: {overall_metrics['non_empty_ratio']:.4f}")
        print("-" * 20)
        for level in ["Easy", "Medium", "Hard"]:
            m = level_metrics[level]
            if m['count'] > 0:
                print(f"{level:<8} (n={m['count']:<3}): EM={m['em']:.4f}, LLM_Eq={m['llm_equal']:.4f}")
        
        print("-" * 20)
        for short_label in CATEGORY_ORDER:
            m = category_metrics[short_label]
            if m["count"] > 0:
                print(f"{short_label:<5} (n={m['count']:<3}): EM={m['em']:.4f}, LLM_Eq={m['llm_equal']:.4f}")
        print(f"{'='*50}")
        
    finally:
        for agent in agents:
             await agent.close()
        if _EVAL_HTTP_CLIENT is not None:
            await _EVAL_HTTP_CLIENT.aclose()


if __name__ == "__main__":
    asyncio.run(main())
