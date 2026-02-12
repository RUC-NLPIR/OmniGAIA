"""
run_omni_atlas.py — Run the OmniAtlas agent on the OmniGAIA benchmark.

OmniAtlas extends the base agent with *active perception* tools (read_video,
read_audio, read_image) that allow the model to request additional media
segments during reasoning.

Environment variables:
    EVAL_BASE_URL:  Base URL of the evaluation LLM endpoint.
    EVAL_API_KEY:   API key for the evaluation endpoint (default: "empty").
    EVAL_MODEL:     Model name used for LLM-based answer equivalence checking.
"""
import os
import json
import asyncio
import logging
import base64
import mimetypes
import httpx
import argparse
import shutil
import subprocess
import re
import time
import math
import tempfile
import random
import cv2
import numpy as np
import io
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

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

# =============================================================================
# Configuration
# =============================================================================

SYSTEM_PROMPT = (
    "You are an omni-modal general AI assistant. Please answer the question "
    "provided to you based on the input image, audio, or video content.\n\n"
    "You should think step by step to answer the question. You may use "
    "available tools to assist with your analysis if needed.\n\n"
    "**Note:**\n"
    '- If there are segments in the input image/audio/video that are unclear '
    'to you, you should use the "read_image/read_audio/read_video" tool to '
    "examine them carefully to ensure you have correctly perceived the input "
    "media.\n\n"
    "Please provide your final answer using this format: "
    "<answer>YOUR_ANSWER</answer>."
)

# Retry Config
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2

# Evaluation LLM (configurable via environment variables)
EVAL_ENDPOINTS = [
    {
        "base_url": os.getenv("EVAL_BASE_URL", "http://localhost:8089/v1"),
        "api_key": os.getenv("EVAL_API_KEY", "empty"),
        "model": os.getenv("EVAL_MODEL", "deepseek-v3"),
    },
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("OmniAtlas")

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

# =============================================================================
# External Tools (web/code)
# =============================================================================

from tools.web_tools import (
    web_search,
    page_browser,
    get_openai_function_web_search,
    get_openai_function_page_browser,
)
from tools.code_executor import (
    code_executor,
    get_openai_function_code_executor,
)


class AgentTools:
    """
    Wrapper for tool implementations.
    For OmniAtlas, the actual logic for media reading and eviction is handled in the agent loop.
    These methods return metadata/confirmation for the model.
    """

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


# =============================================================================
# Tool Schemas (for Hermes-style prompt injection)
# =============================================================================


def get_function_schema_read_video():
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


def get_function_schema_read_audio():
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


def get_function_schema_read_image():
    return {
        "type": "function",
        "function": {
            "name": "read_image",
            "description": "Reads specific images to view them in detail. Optionally crop the image by providing a crop box [left, top, right, bottom].",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_ids": {"type": "array", "items": {"type": "string"}, "description": "List of image identifiers or filenames."},
                    "crop_box": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Optional. A 4-element list [left, top, right, bottom] specifying the cropping rectangle.",
                    },
                },
                "required": ["image_ids"],
            },
        },
    }


def _build_tools_prompt(openai_tools_schema: List[Dict[str, Any]]) -> str:
    """Build Hermes-style tools prompt for manual tool calling (same idea as run_base_agent.py QwenBaseAgent)."""
    tools_json = []
    for tool in openai_tools_schema:
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


def _parse_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse <tool_call> tags from model output (same as run_base_agent.py)."""
    tool_calls = []
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, content or "", re.DOTALL)
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append(
                {
                    "id": f"call_{i}_{int(time.time() * 1000)}",
                    "name": data.get("name"),
                    "arguments": data.get("arguments", {}),
                }
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {match}, error: {e}")
    return tool_calls


def _format_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    """Format tool results for Hermes-style template."""
    results_str = []
    for r in tool_results:
        results_str.append(f"{r['content']}")
    return "\n".join(results_str)


# =============================================================================
# Helper Functions (Resize, etc.)
# =============================================================================


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 1024 * 1024,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
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
# OmniInfoManager
# =============================================================================


class OmniInfoManager:
    """Resolve file paths based on media IDs embedded in questions.

    Set the environment variable ``OMNIGAIA_DATA_DIR`` to point at the root
    directory that contains the benchmark media files.  Expected layout::

        $OMNIGAIA_DATA_DIR/
          video/       # .mp4, .mkv, …
          audio/       # .wav, .mp3, …
          image/       # .jpg, .png, …
    """

    _DATA_ROOT = os.getenv("OMNIGAIA_DATA_DIR", os.path.join(".", "data"))

    BASE_DIRS = {
        "video": [os.path.join(_DATA_ROOT, "video")],
        "audio": os.path.join(_DATA_ROOT, "audio"),
        "image": os.path.join(_DATA_ROOT, "image"),
    }

    EXTENSIONS = {
        "video": [".mp4", ".mkv", ".avi", ".webm"],
        "audio": [".wav", ".mp3", ".flac", ".m4a"],
        "image": [".jpg", ".jpeg", ".png"],
    }

    def _find_file(self, base_dirs: Union[str, List[str]], raw_id: str, extensions: List[str]) -> Optional[str]:
        if isinstance(base_dirs, str):
            base_dirs = [base_dirs]
        for base_dir in base_dirs:
            if os.path.exists(os.path.join(base_dir, raw_id)):
                return os.path.join(base_dir, raw_id)
            for ext in extensions:
                if os.path.exists(os.path.join(base_dir, raw_id + ext)):
                    return os.path.join(base_dir, raw_id + ext)
        return None

    def get_file_path(self, media_id: str) -> Optional[str]:
        clean_id = media_id
        for p in ["video_", "audio_", "image_"]:
            if clean_id.startswith(p):
                clean_id = clean_id[len(p):]

        found = self._find_file(self.BASE_DIRS["video"], clean_id, self.EXTENSIONS["video"])
        if found:
            return found
        found = self._find_file(self.BASE_DIRS["audio"], clean_id, self.EXTENSIONS["audio"])
        if found:
            return found
        found = self._find_file(self.BASE_DIRS["image"], clean_id, self.EXTENSIONS["image"])
        if found:
            return found

        if os.path.exists(media_id):
            return media_id
        return None

    def get_file_paths_for_item(self, item: Dict[str, Any]) -> List[str]:
        paths = []
        ids = re.findall(r"<((?:image|audio|video)_[^>]+)>", item.get("question", ""))
        for mid in ids:
            path = self.get_file_path(mid)
            if path:
                paths.append(path)
            else:
                logger.warning(f"File path not found for Media ID in question: {mid}")
        return list(set(paths))


# =============================================================================
# OmniAtlas Qwen Agent (Hermes manual tool calling)
# =============================================================================


class OmniAtlasQwenAgent:
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
        self.api_base_url = api_base_url
        self.request_timeout = max(30, int(request_timeout))
        self.forced_final_timeout = max(30, int(forced_final_timeout))
        self.ffmpeg_timeout = max(10, int(ffmpeg_timeout))
        self.http_client = httpx.AsyncClient(timeout=3600, trust_env=False, follow_redirects=True)
        self.tools = AgentTools()
        self.info_manager = OmniInfoManager()

        if AsyncOpenAI is None:
            raise ImportError("openai package is required")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base_url,
            http_client=self.http_client,
        )

    async def close(self):
        await self.http_client.aclose()

    # ---------- Media helpers ----------

    def _to_data_url(self, path: str, default_mime: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or default_mime
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return ""

    def _build_part(self, kind: str, url: str) -> Dict[str, Any]:
        """QwenBaseAgent-style media part."""
        return {"type": f"{kind}_url", f"{kind}_url": {"url": url}}

    def _register_media_id(self, media_id: str, path: str, mapping: Dict[str, str]) -> None:
        if not media_id:
            return
        if media_id not in mapping:
            mapping[media_id] = path
        if media_id.startswith(("image_", "audio_", "video_")):
            stripped = media_id.split("_", 1)[1]
            if stripped and stripped not in mapping:
                mapping[stripped] = path

    def _build_media_id_map(self, media_items: List[Union[str, Dict]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for item in media_items:
            path = item.get("path") if isinstance(item, dict) else item
            if not path or not os.path.exists(path):
                continue
            base = os.path.splitext(os.path.basename(path))[0]
            if isinstance(item, dict):
                self._register_media_id(base, path, mapping)
                kind = item.get("type")
                if kind:
                    prefixed = f"{kind}_{base}"
                    self._register_media_id(prefixed, path, mapping)
            else:
                self._register_media_id(base, path, mapping)
        return mapping

    def _resolve_media_path(self, media_id: str, media_id_to_path: Dict[str, str]) -> Optional[str]:
        media_id = self._normalize_media_id(media_id)
        if not media_id:
            return None
        path = media_id_to_path.get(media_id)
        if path:
            return path
        return self.info_manager.get_file_path(media_id)

    def _normalize_media_id(self, media_id: Any) -> str:
        if media_id is None:
            return ""
        if isinstance(media_id, (int, float)):
            return str(media_id).strip()
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

    def _compress_video(self, video_path: str, max_frames: int = 128) -> Optional[Dict[str, Any]]:
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

            target_fps = 2.0
            duration = total_frames / video_fps
            nframes = int(duration * target_fps)
            nframes = min(nframes, max_frames)
            nframes = max(1, nframes)
            sample_indices = np.linspace(0, total_frames - 1, nframes, dtype=int)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None

            height, width = frame.shape[:2]
            new_height, new_width = smart_resize(height, width, factor=16, min_pixels=224 * 224, max_pixels=512 * 512)

            fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = 2.0
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

            comp_size = os.path.getsize(temp_video_path) / (1024 * 1024)
            logger.info(f"Compressed video {os.path.basename(video_path)} to {comp_size:.2f}MB ({len(sample_indices)} frames)")

            with open(temp_video_path, "rb") as f:
                video_data = f.read()
                b64_video = base64.b64encode(video_data).decode("utf-8")

            return {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{b64_video}"},
            }
        except Exception as e:
            logger.error(f"Failed to compress video {video_path}: {e}")
            return None
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception:
                    pass

    def _compress_video_segment(
        self, video_path: str, t_start: float, t_end: float, max_frames: int = 128
    ) -> Optional[Dict[str, Any]]:
        if not os.path.exists(video_path):
            return None

        temp_video_path = None
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if total_frames <= 0 or video_fps <= 0:
                cap.release()
                return None

            start_frame = int(max(0.0, t_start) * video_fps)
            end_frame = int(t_end * video_fps) if t_end is not None else total_frames
            end_frame = min(end_frame, total_frames)
            if start_frame >= end_frame:
                start_frame = max(0, end_frame - 1)

            seg_cnt = end_frame - start_frame
            if seg_cnt <= 0:
                cap.release()
                return None

            duration = seg_cnt / video_fps
            target_fps = min(2.0, max_frames / duration) if duration > 0 else 2.0
            nframes = int(duration * target_fps)
            nframes = min(nframes, max_frames)
            nframes = max(1, nframes)
            sample_indices = np.linspace(start_frame, end_frame - 1, nframes, dtype=int)

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_indices[0]))
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None

            height, width = frame.shape[:2]
            new_height, new_width = smart_resize(
                height, width, factor=28, min_pixels=16 * 16, max_pixels=256 * 256
            )

            fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                try:
                    cmd = [
                        ffmpeg_path,
                        "-y",
                        "-ss",
                        str(max(0.0, t_start)),
                        "-to",
                        str(t_end),
                        "-i",
                        video_path,
                        "-vf",
                        f"fps={target_fps},scale={new_width}:{new_height}",
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a?",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-b:a",
                        "64k",
                        "-shortest",
                        temp_video_path,
                    ]
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.ffmpeg_timeout,
                    )
                    if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
                        with open(temp_video_path, "rb") as f:
                            b64_video = base64.b64encode(f.read()).decode("utf-8")
                        return {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{b64_video}"},
                        }
                except Exception as e:
                    logger.warning(
                        f"ffmpeg video+audio extraction failed, fallback to video-only: {e}"
                    )
                    try:
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                    except Exception:
                        pass
                    fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
                    os.close(fd)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (new_width, new_height))

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                if new_height != height or new_width != width:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                out.write(frame)

            cap.release()
            out.release()

            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                logger.warning(f"Video segment encoding produced empty file for {video_path}")
                return None

            with open(temp_video_path, "rb") as f:
                b64_video = base64.b64encode(f.read()).decode("utf-8")
            return {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{b64_video}"},
            }
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
        if not AudioSegment:
            return None
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            start_ms = t_start * 1000
            end_ms = t_end * 1000 if t_end is not None else len(audio)
            segment = audio[start_ms:end_ms]

            if len(segment) > 300 * 1000:
                segment = segment[: 300 * 1000]

            buf = io.BytesIO()
            segment.export(buf, format="mp3")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}}
        except Exception as e:
            logger.error(f"Audio compression failed: {e}")
            return None

    def _get_initial_media_parts(self, media_items: List[Union[str, Dict]]) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
        parts: List[Dict[str, Any]] = []
        tool_flags = {"read_video": False, "read_audio": False, "read_image": False}

        frames_per_video = 128
        image_count = 0

        for item in media_items:
            path = item.get("path") if isinstance(item, dict) else item
            if not path or not os.path.exists(path):
                continue

            media_id = os.path.splitext(os.path.basename(path))[0]

            kind = "image"
            if path.endswith((".mp4", ".mkv", ".avi", ".webm")):
                kind = "video"
            elif path.endswith((".wav", ".mp3", ".flac", ".m4a")):
                kind = "audio"

            if kind == "video":
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                file_size = os.path.getsize(path)
                parts.append(
                    {
                        "type": "text",
                        "text": f"Media ID: {media_id}\nMedia Type: Video\nDuration: {duration:.2f}s\nResolution: {width}x{height}\nSize: {file_size/1024/1024:.2f}MB",
                    }
                )

                tool_flags["read_video"] = True

                logger.info(
                    f"Processing video {os.path.basename(path)} with LlamaFactory strategy (2FPS, Max 128 frames, 256x256)."
                )
                compressed_video_part = self._compress_video(path, max_frames=frames_per_video)
                if compressed_video_part:
                    parts.append(compressed_video_part)

                # Extract Audio (First 15 mins) if pydub available
                if AudioSegment:
                    try:
                        audio = AudioSegment.from_file(path)
                        audio = audio.set_channels(1)
                        audio = audio.set_frame_rate(16000)
                        fifteen_mins_ms = 15 * 60 * 1000
                        if len(audio) > fifteen_mins_ms:
                            audio = audio[:fifteen_mins_ms]
                        buf = io.BytesIO()
                        audio.export(buf, format="mp3")
                        b64_audio = base64.b64encode(buf.getvalue()).decode("utf-8")
                        parts.append({"type": "input_audio", "input_audio": {"data": b64_audio, "format": "mp3"}})
                        logger.info(
                            f"Extracted audio from {os.path.basename(path)} (Size: {len(b64_audio)/(1024*1024):.2f}MB base64)"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to extract audio from video {path}: {e}")

            elif kind == "audio":
                file_size = os.path.getsize(path)
                duration_str = "Unknown"
                if AudioSegment:
                    try:
                        audio_check = AudioSegment.from_file(path)
                        duration_s = len(audio_check) / 1000.0
                        duration_str = f"{duration_s:.2f}s"
                    except Exception:
                        pass

                parts.append(
                    {
                        "type": "text",
                        "text": f"Media ID: {media_id}\nMedia Type: Audio\nDuration: {duration_str}\nSize: {file_size/1024/1024:.2f}MB",
                    }
                )

                b64_data = None
                fmt = "wav"
                if AudioSegment:
                    try:
                        audio = AudioSegment.from_file(path)
                        audio = audio.set_channels(1)
                        audio = audio.set_frame_rate(16000)

                        tool_flags["read_audio"] = True

                        fifteen_mins_ms = 15 * 60 * 1000
                        if len(audio) > fifteen_mins_ms:
                            logger.info(f"Audio {os.path.basename(path)} > 15m. Truncating to 15m.")
                            audio = audio[:fifteen_mins_ms]

                        buf = io.BytesIO()
                        audio.export(buf, format="mp3")
                        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                        fmt = "mp3"
                    except Exception as e:
                        logger.warning(f"Pydub processing failed for {path}: {e}")

                if not b64_data:
                    try:
                        max_audio_size = 20 * 1024 * 1024
                        file_size = os.path.getsize(path)
                        with open(path, "rb") as f:
                            if file_size > max_audio_size:
                                logger.info(
                                    f"Audio {os.path.basename(path)} is large ({file_size/(1024*1024):.2f}MB). Binary truncating to 20MB."
                                )
                                audio_data = f.read(max_audio_size)
                            else:
                                audio_data = f.read()
                        b64_data = base64.b64encode(audio_data).decode("utf-8")
                        if file_size > 5 * 1024 * 1024:
                            tool_flags["read_audio"] = True
                    except Exception as e:
                        logger.error(f"Failed to process audio {path}: {e}")

                if b64_data:
                    parts.append({"type": "input_audio", "input_audio": {"data": b64_data, "format": fmt}})

            else:
                image_count += 1
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                file_size = os.path.getsize(path)
                parts.append(
                    {
                        "type": "text",
                        "text": f"Media ID: {media_id}\nMedia Type: Image\nResolution: {w}x{h}\nSize: {file_size/1024/1024:.2f}MB",
                    }
                )
                url = self._to_data_url(path, "image/jpeg")
                parts.append(self._build_part("image", url))

        tool_flags["read_image"] = True
        return parts, tool_flags

    # ---------- Tool execution (active perception) ----------

    async def _handle_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        messages: List[Dict[str, Any]],
        media_id_to_path: Dict[str, str],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes tool and returns (tool_result_text, media_parts_to_attach).
        For read_xxx tools, media parts are returned to be attached into the same role=tool message,
        instead of being injected as extra role=user messages.
        Note: tool calls may attach media parts to the tool response message.
        """
        tool_output_content = ""
        media_parts: List[Dict[str, Any]] = []

        if tool_name == "read_video":
            vid = self._normalize_media_id(tool_args.get("video_id"))
            t_start = self._parse_time_value(tool_args.get("t_start")) or 0.0
            t_end = self._parse_time_value(tool_args.get("t_end"))
            path = self._resolve_media_path(vid, media_id_to_path)
            if path:
                cap = cv2.VideoCapture(path)
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                file_duration = total_frames / fps if fps > 0 else 0
                cap.release()

                real_end = t_end if t_end is not None else file_duration
                seg_duration = real_end - t_start

                video_segment = self._compress_video_segment(path, t_start=t_start, t_end=t_end)
                if video_segment:
                    segment_id = f"{vid}_segment_{t_start}_{t_end}"
                    tool_output_content = json.dumps(
                        {"status": "success", "message": f"Video segment loaded. Assigned Media ID: {segment_id}"},
                        ensure_ascii=False,
                    )

                    media_info = (
                        f"Media ID: {segment_id}\n"
                        f"Media Type: Video Segment\n"
                        f"Segment Duration: {seg_duration:.2f}s (from {t_start}s to {real_end}s)\n"
                        f"Original Resolution: {orig_w}x{orig_h}"
                    )
                    media_parts.extend([{"type": "text", "text": media_info}, video_segment])
                else:
                    tool_output_content = json.dumps({"status": "error", "message": "Could not extract video segment."})
            else:
                tool_output_content = json.dumps({"status": "error", "message": f"Video {vid} not found."})

        elif tool_name == "read_audio":
            aid = self._normalize_media_id(tool_args.get("audio_id"))
            t_start = self._parse_time_value(tool_args.get("t_start")) or 0.0
            t_end = self._parse_time_value(tool_args.get("t_end"))
            path = self._resolve_media_path(aid, media_id_to_path)
            if path:
                seg_duration_str = "Unknown"
                if AudioSegment:
                    try:
                        audio = AudioSegment.from_file(path)
                        total_len_s = len(audio) / 1000.0
                        real_end = t_end if t_end is not None else total_len_s
                        seg_duration = real_end - t_start
                        seg_duration_str = f"{seg_duration:.2f}s (from {t_start}s to {real_end}s)"
                    except Exception:
                        pass

                part = self._compress_audio_segment(path, t_start, t_end)
                if part:
                    segment_id = f"{aid}_segment_{t_start}_{t_end}"
                    tool_output_content = json.dumps(
                        {"status": "success", "message": f"Audio segment loaded. Assigned Media ID: {segment_id}"},
                        ensure_ascii=False,
                    )
                    media_info = f"Media ID: {segment_id}\nMedia Type: Audio Segment\nSegment Duration: {seg_duration_str}"
                    media_parts.extend([{"type": "text", "text": media_info}, part])
                else:
                    tool_output_content = json.dumps({"status": "error", "message": "Could not extract audio."})
            else:
                tool_output_content = json.dumps({"status": "error", "message": f"Audio {aid} not found."})

        elif tool_name == "read_image":
            iids = tool_args.get("image_ids", [])
            crop_box = tool_args.get("crop_box")
            loaded_ids = []

            for iid in iids:
                path = self._resolve_media_path(iid, media_id_to_path)
                if not path:
                    continue
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                if crop_box and len(crop_box) == 4:
                    x1, y1, x2, y2 = crop_box
                    x1 = max(0, min(int(x1), w))
                    y1 = max(0, min(int(y1), h))
                    x2 = max(0, min(int(x2), w))
                    y2 = max(0, min(int(y2), h))
                    if x2 > x1 and y2 > y1:
                        img = img[y1:y2, x1:x2]
                        h, w = img.shape[:2]
                        detail_id = f"{iid}_crop_{x1}_{y1}_{x2}_{y2}"
                        type_desc = f"Image Crop (Box: {x1},{y1},{x2},{y2})"
                    else:
                        detail_id = f"{iid}_detail"
                        type_desc = "Image Detail"
                else:
                    detail_id = f"{iid}_detail"
                    type_desc = "Image Detail"

                _, buf = cv2.imencode(".jpg", img)
                b64 = base64.b64encode(buf).decode("utf-8")
                url = f"data:image/jpeg;base64,{b64}"
                data_size = len(b64)

                loaded_ids.append(detail_id)
                media_info = (
                    f"Media ID: {detail_id}\nMedia Type: {type_desc}\nResolution: {w}x{h}\nSize: {data_size/1024/1024:.2f}MB (base64)"
                )
                media_parts.extend([{"type": "text", "text": media_info}, self._build_part("image", url)])

            if loaded_ids:
                tool_output_content = json.dumps(
                    {"status": "success", "message": f"Images loaded. Assigned Media IDs: {loaded_ids}"},
                    ensure_ascii=False,
                )
            else:
                tool_output_content = json.dumps({"status": "error", "message": "No images found."})

        elif tool_name in ["web_search", "page_browser", "code_executor"]:
            if hasattr(self.tools, tool_name):
                tool_func = getattr(self.tools, tool_name)
                max_tool_retries = 3
                for attempt in range(max_tool_retries):
                    try:
                        result = await tool_func(**tool_args)
                        if not result and attempt < max_tool_retries - 1:
                            logger.warning(f"Tool {tool_name} returned empty result (Attempt {attempt+1}). Retrying...")
                            await asyncio.sleep(1)
                            continue
                        tool_output_content = json.dumps(result, ensure_ascii=False)
                        break
                    except Exception as e:
                        if attempt < max_tool_retries - 1:
                            logger.warning(f"Tool {tool_name} failed (Attempt {attempt+1}): {e}. Retrying...")
                            await asyncio.sleep(1)
                        else:
                            tool_output_content = json.dumps({"error": str(e)}, ensure_ascii=False)
            else:
                tool_output_content = json.dumps({"error": f"Tool {tool_name} implementation not found."}, ensure_ascii=False)
        else:
            tool_output_content = json.dumps({"status": "error", "message": f"Tool {tool_name} not found."}, ensure_ascii=False)

        return tool_output_content, media_parts

    # ---------- Qwen chat loop ----------

    async def run(self, question: str, media_items: List[Union[str, Dict]] = []) -> Dict[str, Any]:
        logger.info(f"Starting OmniAtlas Qwen Agent ({self.model})")

        media_id_to_path = self._build_media_id_map(media_items)
        content_parts, tool_flags = self._get_initial_media_parts(media_items)
        if question:
            content_parts.append({"type": "text", "text": question})

        # Tool list dynamically (same as yinli)
        openai_tools_schema: List[Dict[str, Any]] = []
        if tool_flags.get("read_video", False):
            openai_tools_schema.append(get_function_schema_read_video())
        if tool_flags.get("read_audio", False):
            openai_tools_schema.append(get_function_schema_read_audio())
        if tool_flags.get("read_image", False):
            openai_tools_schema.append(get_function_schema_read_image())

        openai_tools_schema.append(get_openai_function_web_search())
        openai_tools_schema.append(get_openai_function_page_browser())
        openai_tools_schema.append(get_openai_function_code_executor())

        system_content = SYSTEM_PROMPT + "\n\n" + _build_tools_prompt(openai_tools_schema)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_content}]},
            {"role": "user", "content": content_parts},
        ]

        max_turns = 50
        all_tool_calls: List[Dict[str, Any]] = []

        for turn in range(max_turns):
            logger.info(f"Turn {turn+1}...")

            async def _do_request_with_retry():
                current_max_tokens = 10240
                for attempt in range(MAX_RETRIES):
                    try:
                        return await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=current_max_tokens,
                            timeout=self.request_timeout,
                            temperature=0.6,
                            top_p=0.95,
                            extra_body={
                                "top_k": 20,
                                "repetition_penalty": 1.05,
                            },
                        )
                    except Exception as e:
                        err_str = str(e)
                        logger.error(f"API Request failed (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
                        if (
                            "max_tokens" in err_str
                            and "too large" in err_str
                            and attempt < MAX_RETRIES - 1
                        ):
                            new_max_tokens = max(256, current_max_tokens // 2)
                            if new_max_tokens < current_max_tokens:
                                logger.info(
                                    f"Reducing max_tokens from {current_max_tokens} to {new_max_tokens} and retrying."
                                )
                                current_max_tokens = new_max_tokens
                                await asyncio.sleep(RETRY_DELAY_BASE * (2**attempt))
                                continue
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY_BASE * (2**attempt))
                        else:
                            raise

            try:
                response = await _do_request_with_retry()
            except Exception as e:
                logger.error(f"Error in request: {e}")
                return {"output": f"Error: {e}", "messages": messages, "tool_calls": all_tool_calls}

            content = response.choices[0].message.content or ""
            reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning_content:
                content = f"<think>{reasoning_content}</think>\n" + content

            # Append assistant message
            messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

            # Parse tool calls
            tool_calls = _parse_tool_calls(content)
            if not tool_calls:
                if content:
                    logger.info(f"Final Answer: {content[:200]}...")
                    return {"output": content, "tool_calls": all_tool_calls, "messages": messages}
                return {"output": "Error: Empty response", "tool_calls": all_tool_calls, "messages": messages}

            all_tool_calls.extend(tool_calls)

            # Execute tools in parallel
            async def execute_tool(tc: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
                fn_name = tc.get("name")
                fn_args = tc.get("arguments", {})

                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        pass

                logger.info(f"Executing tool: {fn_name} with args: {fn_args}")

                if not fn_name:
                    tool_content = json.dumps({"status": "error", "message": "Missing tool name"}, ensure_ascii=False)
                    return {"name": "unknown", "content": tool_content}, []

                max_tool_retries = 3
                last_err = None
                for attempt in range(max_tool_retries):
                    try:
                        tool_result_content, media_parts = await self._handle_tool_execution(
                            fn_name, fn_args, messages, media_id_to_path
                        )
                        wrapped = json.dumps(
                            {"tool": fn_name, "result": json.loads(tool_result_content) if tool_result_content else {}},
                            ensure_ascii=False,
                        )
                        return {"name": fn_name, "content": wrapped}, media_parts
                    except Exception as e:
                        last_err = e
                        if attempt < max_tool_retries - 1:
                            logger.warning(f"Tool {fn_name} failed (Attempt {attempt+1}): {e}. Retrying...")
                            await asyncio.sleep(1)
                        else:
                            wrapped = json.dumps({"tool": fn_name, "error": str(last_err)}, ensure_ascii=False)
                            return {"name": fn_name, "content": wrapped}, []

            exec_results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])
            # Build ONE role=tool message that interleaves:
            # (text) + media parts returned by read_xxx tools.
            tool_content_parts: List[Dict[str, Any]] = []
            for tool_result, media_parts in exec_results:
                tool_content_parts.append(
                    {"type": "text", "text": f"{tool_result['content']}"}
                )
                if media_parts:
                    tool_content_parts.extend(media_parts)

            if not tool_content_parts:
                tool_content_parts = [{"type": "text", "text": "{}"}]

            messages.append({"role": "tool", "content": tool_content_parts})

        logger.info(f"Max turns ({max_turns}) reached. Forcing final answer without tools.")
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You have exceeded the maximum number of turns ({max_turns}). You are not allowed to call tools anymore. Please provide your final answer to the user's question now based on the information you have.",
                    }
                ],
            }
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=10240,
                timeout=self.forced_final_timeout,
            )
            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
            return {"output": content or "Error: Empty response from model after forced generation.", "tool_calls": all_tool_calls, "messages": messages}
        except Exception as e:
            logger.error(f"Error during forced final response: {e}")
            return {"output": f"Error: {str(e)}", "tool_calls": all_tool_calls, "messages": messages}


# =============================================================================
# Helper: Check Equivalence (same logic as run_omni_atlas_yinli.py)
# =============================================================================


async def check_equivalence(
    question: str,
    predicted: str,
    standard: str,
    request_timeout: int = 120,
    max_tokens: int = 2048,
) -> Tuple[bool, str]:
    if not predicted:
        return False, "No prediction provided."

    prompt = f"""Please determine if the model correctly predicted the answer.
Question: {question}
Model Predicted Answer: {predicted}
Labeled Answer: {standard}
Return 'Correct' if the model's prediction is completely accurate, otherwise return 'Incorrect'. Provide only this single word response."""

    if AsyncOpenAI is None:
        raise ImportError("openai package is required for AsyncOpenAI client")

    endpoint = random.choice(EVAL_ENDPOINTS)
    request_timeout = max(10, int(request_timeout))
    max_tokens = max(32, int(max_tokens))
    http_client = httpx.AsyncClient(timeout=max(120, request_timeout + 30), trust_env=False)
    client = AsyncOpenAI(
        api_key=endpoint["api_key"],
        base_url=endpoint["base_url"],
        http_client=http_client,
    )

    try:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=endpoint["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"thinking": False}},
                    timeout=request_timeout,
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
                    await asyncio.sleep(RETRY_DELAY_BASE * (2**attempt))
    finally:
        await http_client.aclose()

    return False, f"Eval Request Failed after {MAX_RETRIES} attempts."


# =============================================================================
# Helper: Metrics (same as run_omni_atlas_yinli.py)
# =============================================================================


def get_modality_category(item: Dict[str, Any]) -> str:
    omni_input = item.get("omni_modal_input", [])
    if not isinstance(omni_input, list):
        return "audio_image"
    if any(isinstance(inp, dict) and inp.get("type") == "video" for inp in omni_input):
        return "video"
    return "audio_image"


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"count": 0, "em": 0.0, "llm_equal": 0.0, "avg_tool_calls": 0.0, "non_empty_ratio": 0.0}

    count = len(results)
    total_em = sum(r.get("em_score", 0) for r in results)
    avg_em = total_em / count

    def is_correct(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val > 0
        return False

    total_llm = sum(1 for r in results if is_correct(r.get("llm_equal")))
    avg_llm = total_llm / count

    total_tool_calls = sum(r.get("tool_call_num", 0) for r in results)
    avg_tool_calls = total_tool_calls / count

    non_empty = sum(1 for r in results if r.get("predicted_answer"))
    non_empty_ratio = non_empty / count

    return {
        "count": count,
        "em": avg_em,
        "llm_equal": avg_llm,
        "avg_tool_calls": avg_tool_calls,
        "non_empty_ratio": non_empty_ratio,
    }


def calculate_category_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    category_metrics: Dict[str, Any] = {}
    for short_label in CATEGORY_ORDER:
        full_label = CATEGORY_LABEL_MAP.get(short_label, short_label)
        cat_items = [r for r in results if r.get("category") == full_label]
        category_metrics[short_label] = calculate_metrics(cat_items)
    return category_metrics


# =============================================================================
# Main Execution (same output format as run_omni_atlas_yinli.py)
# =============================================================================


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_base_url", type=str, required=True)
    parser.add_argument("--level", type=str, default=None, help="Filter items by difficulty level (Easy, Medium, Hard)")
    parser.add_argument("--max_items", type=int, default=None, help="Limit the number of items to process.")
    parser.add_argument("--concurrent_limit", type=int, default=5, help="Maximum number of concurrent API calls")
    parser.add_argument("--request_timeout", type=int, default=600, help="Per-request timeout (seconds) for main model calls.")
    parser.add_argument(
        "--forced_final_timeout",
        type=int,
        default=300,
        help="Timeout (seconds) for forced final answer call after max tool turns.",
    )
    parser.add_argument("--ffmpeg_timeout", type=int, default=180, help="Timeout (seconds) for ffmpeg subprocess.")
    parser.add_argument("--item_timeout", type=int, default=1800, help="Max total processing time (seconds) per item.")
    parser.add_argument("--eval_timeout", type=int, default=120, help="Timeout (seconds) for equivalence evaluation calls.")
    parser.add_argument("--eval_max_tokens", type=int, default=2048, help="Max tokens for equivalence evaluation.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip LLM-based equivalence evaluation.")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return

    agent = OmniAtlasQwenAgent(
        args.api_key,
        args.model_name,
        args.api_base_url,
        request_timeout=args.request_timeout,
        forced_final_timeout=args.forced_final_timeout,
        ffmpeg_timeout=args.ffmpeg_timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrent_limit)

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            items = json.load(f)
            if not isinstance(items, list):
                items = [items]

        if args.level:
            target_level = args.level.strip().capitalize()
            original_count = len(items)
            items = [item for item in items if item.get("Level") == target_level]
            print(f"Filtered items by level '{target_level}': {len(items)}/{original_count} kept.")

        if args.max_items:
            items = items[: args.max_items]

        def _build_failed_result(item: Dict[str, Any], question_text: str, reason: str) -> Dict[str, Any]:
            return {
                "id": item.get("id"),
                "question": question_text,
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
                "messages": [],
                "tool_call_num": 0,
                "predicted_answer": "",
                "em_score": 0,
                "llm_equal": 0,
                "llm_eval_response": reason,
            }

        async def _process_item_inner(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            q = item.get("question", "")
            if not q:
                return None

            media: List[Union[str, Dict[str, Any]]] = []
            if "omni_modal_input" in item and item["omni_modal_input"]:
                for inp in item["omni_modal_input"]:
                    if isinstance(inp, dict) and "path" in inp:
                        media.append(inp)
            else:
                paths = agent.info_manager.get_file_paths_for_item(item)
                media = paths

            if "file_input" in item:
                file_paths = []
                for fobj in item["file_input"]:
                    if isinstance(fobj, dict) and "path" in fobj:
                        file_paths.append(fobj["path"])
                if file_paths:
                    q += "\n\nReferenced Files:\n" + "\n".join(file_paths)

            res = await agent.run(q, media)
            messages = res.get("messages", [])

            def _truncate_large_data(obj):
                if isinstance(obj, dict):
                    new_obj = {}
                    for k, v in obj.items():
                        if k == "url" and isinstance(v, str) and v.startswith("data:") and len(v) > 1000:
                            new_obj[k] = "<data_url_truncated>"
                        elif k == "data" and isinstance(v, str) and len(v) > 1000:
                            new_obj[k] = "<base64_data_truncated>"
                        else:
                            new_obj[k] = _truncate_large_data(v)
                    return new_obj
                if isinstance(obj, list):
                    return [_truncate_large_data(x) for x in obj]
                return obj

            truncated_messages = _truncate_large_data(messages)

            predicted_answer = res.get("output", "")
            matches = re.findall(r"<answer>(.*?)</answer>", predicted_answer, re.DOTALL)
            if matches:
                final_answer = matches[-1].strip()
            else:
                words = predicted_answer.split()
                final_answer = " ".join(words[-20:])

            eval_predicted_text = final_answer
            if not eval_predicted_text:
                cleaned_output = re.sub(r"<think>.*?</think>", "", predicted_answer, flags=re.DOTALL).strip()
                eval_predicted_text = cleaned_output

            def normalize_text(s):
                if not s:
                    return ""
                s = str(s).lower()
                s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                return " ".join(s.split())

            ground_truth = item.get("answer", "")
            em_score = 1 if normalize_text(final_answer) == normalize_text(ground_truth) else 0

            if final_answer:
                if em_score == 1:
                    llm_equal = True
                    llm_eval_response = "EM is 1, skipping LLM evaluation."
                elif args.skip_eval:
                    llm_equal = False
                    llm_eval_response = "Skipped LLM evaluation (--skip_eval)."
                else:
                    llm_equal, llm_eval_response = await check_equivalence(
                        q,
                        eval_predicted_text,
                        ground_truth,
                        request_timeout=args.eval_timeout,
                        max_tokens=args.eval_max_tokens,
                    )
            else:
                llm_equal = False
                llm_eval_response = "Predicted answer is empty, skipping LLM evaluation."

            llm_score = 1 if llm_equal else 0

            tool_calls = res.get("tool_calls", [])
            tool_call_num = len(tool_calls) if isinstance(tool_calls, list) else 0

            return {
                "id": item.get("id"),
                "question": q,
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
                "messages": truncated_messages,
                "tool_call_num": tool_call_num,
                "predicted_answer": final_answer,
                "em_score": em_score,
                "llm_equal": llm_score,
                "llm_eval_response": llm_eval_response,
            }

        async def process_item(item, semaphore):
            async with semaphore:
                original_q = item.get("question", "")
                if not original_q:
                    return None
                try:
                    if args.item_timeout and args.item_timeout > 0:
                        return await asyncio.wait_for(_process_item_inner(item), timeout=args.item_timeout)
                    return await _process_item_inner(item)
                except asyncio.TimeoutError:
                    logger.error(f"Item {item.get('id')} timed out after {args.item_timeout}s")
                    return _build_failed_result(item, original_q, f"Item processing timed out after {args.item_timeout}s.")
                except Exception as e:
                    logger.exception(f"Item {item.get('id')} failed with exception: {e}")
                    return _build_failed_result(item, original_q, f"Item processing failed: {e}")

        print(f"Processing {len(items)} items with concurrency limit {args.concurrent_limit}...")
        tasks = [process_item(item, semaphore) for item in items]

        if tqdm:
            results = []
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                results.append(await f)
        else:
            results = await asyncio.gather(*tasks)

        results = [r for r in results if r is not None]

        overall_metrics = calculate_metrics(results)

        level_metrics = {}
        for level in ["Easy", "Medium", "Hard"]:
            level_results = [r for r in results if r.get("Level") == level]
            level_metrics[level] = calculate_metrics(level_results)

        category_metrics = calculate_category_metrics(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = args.model_name
        output_dir = os.path.join(args.output_dir, f"run_omni_atlas_{model_name_safe}")
        os.makedirs(output_dir, exist_ok=True)

        avg_em = overall_metrics["em"]
        avg_llm_equal = overall_metrics["llm_equal"]
        output_filename = f"run_{timestamp}_em{avg_em:.4f}_llmeq{avg_llm_equal:.4f}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        metrics_data = {"overall": overall_metrics, "by_level": level_metrics, "by_category": category_metrics}
        metrics_filename = f"{os.path.splitext(output_filename)[0]}_metrics.json"
        metrics_path = os.path.join(output_dir, metrics_filename)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

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
            if m["count"] > 0:
                print(f"{level:<8} (n={m['count']:<3}): EM={m['em']:.4f}, LLM_Eq={m['llm_equal']:.4f}")

        print("-" * 20)
        for short_label in CATEGORY_ORDER:
            m = category_metrics[short_label]
            if m["count"] > 0:
                print(f"{short_label:<5} (n={m['count']:<3}): EM={m['em']:.4f}, LLM_Eq={m['llm_equal']:.4f}")
        print(f"{'='*50}")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
