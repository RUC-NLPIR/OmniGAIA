"""
Central configuration loader for OmniGAIA.

Configuration is read from ``config/config.json`` under the project root.
All relative paths in the config file are resolved against the project root.
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


DEFAULT_CONFIG: Dict[str, Any] = {
    "evaluation": {
        "base_url": "http://localhost:8089/v1",
        "api_key": "empty",
        "model": "deepseek-v3",
    },
    "web_tools": {
        "serper_api_key": "",
        "jina_api_key": "",
    },
    "paths": {
        "data_root": "./data",
        "image_save_dir": "./cache/searched_images",
        "web_cache_dir": "./cache",
    },
    "data": {
        "media_dirs": {
            "video": ["videos", "video"],
            "audio": ["audios", "audio"],
            "image": ["images", "image"],
        }
    },
    "agent": {
        "input_file": None,
        "api_key": None,
        "api_base_url": None,
        "model_name": None,
        "level": None,
        "max_items": None,
        "concurrent_limit": 5,
        "use_asr": False,
        "output_dir": "./outputs",
        "request_timeout": 600,
        "forced_final_timeout": 300,
        "ffmpeg_timeout": 180,
        "item_timeout": 1800,
        "eval_timeout": 120,
        "skip_eval": False,
    },
}


_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_CACHE_PATH: Optional[Path] = None


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(path_str: str) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


def _normalize_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths_cfg = cfg.setdefault("paths", {})

    for key in ["data_root", "image_save_dir", "web_cache_dir"]:
        value = paths_cfg.get(key)
        if isinstance(value, str) and value.strip():
            paths_cfg[key] = _resolve_path(value)

    data_cfg = cfg.setdefault("data", {})
    media_cfg = data_cfg.setdefault("media_dirs", {})
    data_root = paths_cfg.get("data_root", _resolve_path("./data"))

    resolved_media_dirs: Dict[str, Any] = {}
    for media_type in ["video", "audio", "image"]:
        raw_dirs = media_cfg.get(media_type, [])
        if isinstance(raw_dirs, str):
            raw_dirs = [raw_dirs]
        if not isinstance(raw_dirs, list):
            raw_dirs = []

        resolved = []
        for d in raw_dirs:
            if not isinstance(d, str) or not d.strip():
                continue
            dir_path = Path(d)
            if not dir_path.is_absolute():
                dir_path = Path(data_root) / d
            resolved.append(str(dir_path.resolve()))
        resolved_media_dirs[media_type] = resolved

    data_cfg["resolved_media_dirs"] = resolved_media_dirs
    return cfg


def load_config(config_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    global _CONFIG_CACHE, _CONFIG_CACHE_PATH

    target_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH.resolve()

    if not force_reload and _CONFIG_CACHE is not None and _CONFIG_CACHE_PATH == target_path:
        return copy.deepcopy(_CONFIG_CACHE)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        if not isinstance(user_cfg, dict):
            raise ValueError(f"Config file must contain a JSON object: {target_path}")
        cfg = _deep_merge_dict(cfg, user_cfg)

    cfg = _normalize_paths(cfg)
    _CONFIG_CACHE = cfg
    _CONFIG_CACHE_PATH = target_path
    return copy.deepcopy(cfg)


def get_config() -> Dict[str, Any]:
    return load_config()

