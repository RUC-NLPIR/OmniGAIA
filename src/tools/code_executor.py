"""
Code Executor Tool - Execute Python code in a sandboxed environment.

Provides a safe runtime that blocks dangerous operations (subprocess, file
deletion, network sockets, etc.) while exposing common scientific and data
processing libraries.

Configuration file:
    config/config.json
"""
import os
import sys
import io
import re
import asyncio
import string
import time
import uuid
from typing import Any, Dict
from contextlib import redirect_stdout
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from config_loader import get_config

logger = logging.getLogger(__name__)

# File output directory (loaded from config/config.json)
APP_CONFIG = get_config()
FILES_OUTPUT_DIR = APP_CONFIG["paths"]["code_files_output_dir"]

# Unsafe patterns to block
UNSAFE_PATTERNS = [
    # Block dangerous system operations
    r'(?<!\w)(subprocess|multiprocessing|threading|ctypes|_thread)\s*[.\(]',
    r'os\.(system|popen|fork|kill|remove|rmdir|unlink)\s*\(',
    r'shutil\.(rmtree|move)\s*\(',
    # Block dangerous built-in functions (but allow exit in specific contexts)
    r'(?<!\w)(eval|exec|__import__|compile)\s*\(',
    # Block network operations that could be dangerous
    r'socket\.socket\s*\(',
]

# Allowed imports for file creation and computation
ALLOWED_IMPORTS = """
import numpy as np
import pandas as pd
import sympy
import math
import json
import csv
import datetime
import random
import string
import re
import collections
import itertools
import openpyxl
import pyarrow
import fastparquet
import h5py
from PIL import Image
import fpdf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import yaml
import sqlite3
import pickle
import logging
import tarfile
import gzip
import zipfile
import wave
import xml.etree.ElementTree as ET
from sympy import symbols, Eq, solve, simplify, expand, factor
from collections import Counter, defaultdict
from itertools import combinations, permutations
from functools import reduce
"""


class CodeExecutionError(Exception):
    """Custom exception for code execution errors"""
    pass


class SafeRuntime:
    """Safe runtime environment for code execution"""
    
    def __init__(self):
        self._global_vars = {}
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup the execution environment with allowed modules"""
        # Execute allowed imports
        exec(ALLOWED_IMPORTS, self._global_vars)
        
        # Add some useful constants
        self._global_vars['FILES_OUTPUT_DIR'] = FILES_OUTPUT_DIR
        
        # Add helper function for generating unique filenames
        self._global_vars['generate_unique_filename'] = generate_unique_filename
    
    def _check_safety(self, code: str) -> None:
        """Check if code contains unsafe patterns"""
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, code):
                raise CodeExecutionError(
                    f"Unsafe code pattern detected. Execution blocked for security reasons."
                )
    
    def exec_code(self, code: str) -> None:
        """Execute code in the safe environment"""
        self._check_safety(code)
        exec(code, self._global_vars)
    
    def eval_code(self, expr: str) -> Any:
        """Evaluate an expression"""
        return eval(expr, self._global_vars)


def generate_unique_filename(extension: str, prefix: str = "") -> str:
    """
    Generate a guaranteed unique filename using UUID.
    
    Args:
        extension: File extension (e.g., 'csv', 'json', '.txt')
        prefix: Optional prefix for the filename
    
    Returns:
        Unique filename like: 1208143025_prefix_a1b2c3d4.csv
        or: 1208143025_a1b2c3d4.csv (without prefix)
    """
    timestamp = time.strftime("%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    ext = extension.lstrip('.')
    
    if prefix:
        return f"{timestamp}_{prefix}_{unique_id}.{ext}"
    return f"{timestamp}_{unique_id}.{ext}"


async def code_executor(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a safe sandbox environment.
    
    This tool can be used for:
    1. Complex mathematical computations that cannot be done mentally
    2. Data processing and analysis
    3. Creating various file types (csv, txt, py, json, etc.)
    
    Args:
        code: Python code to execute. Use print() to output results.
              For file creation, use FILES_OUTPUT_DIR variable directly with standard Python file operations.
              Example: pd.to_csv(os.path.join(FILES_OUTPUT_DIR, "data.csv"), index=False)
              
    Returns:
        Dictionary containing:
        - stdout: Captured print output
        - result: Return value (if any)
        - created_files: List of files created (if any)
        - error: Error message (if execution failed)
    """
    # Ensure output directory exists
    os.makedirs(FILES_OUTPUT_DIR, exist_ok=True)
    
    result = {
        "stdout": "",
        "result": None,
        "created_files": [],
        "error": None,
    }
    
    try:
        runtime = SafeRuntime()
        
        # Clean up code
        code = code.strip()
        
        # Inject FILES_OUTPUT_DIR constant for direct file creation
        full_code = f'FILES_OUTPUT_DIR = "{FILES_OUTPUT_DIR}"\n' + code
        
        # Capture stdout
        stdout_capture = io.StringIO()
        
        def _execute():
            with redirect_stdout(stdout_capture):
                runtime.exec_code(full_code)
        
        # Execute with timeout
        try:
            await asyncio.wait_for(
                asyncio.to_thread(_execute),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            result["error"] = "Code execution timed out after 30 seconds"
            return result
        
        # Get stdout
        stdout_capture.seek(0)
        result["stdout"] = stdout_capture.read().strip()
        
        # Check for created files
        if os.path.exists(FILES_OUTPUT_DIR):
            # Get recently modified files (within last minute)
            current_time = time.time()
            for filename in os.listdir(FILES_OUTPUT_DIR):
                filepath = os.path.join(FILES_OUTPUT_DIR, filename)
                if os.path.isfile(filepath):
                    mtime = os.path.getmtime(filepath)
                    if current_time - mtime < 60:  # Created within last minute
                        result["created_files"].append({
                            "path": filepath,
                            "filename": filename,
                            "size": os.path.getsize(filepath),
                        })
        
        # Try to get the last expression result
        try:
            code_lines = code.strip().split('\n')
            if code_lines:
                last_line = code_lines[-1].strip()
                # Check if last line is an expression (not assignment, not print, etc.)
                if (last_line and 
                    not last_line.startswith(('import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', 'with', '#')) and
                    '=' not in last_line.split('(')[0] and
                    not last_line.endswith(':')):
                    try:
                        expr_result = runtime.eval_code(last_line)
                        if expr_result is not None:
                            result["result"] = str(expr_result)
                    except:
                        pass
        except:
            pass
        
    except CodeExecutionError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = f"Execution error: {str(e)}"
    
    return result


def get_openai_function_code_executor(create_files: bool = False) -> dict:
    """Return the OpenAI tool/function definition for code_executor."""
    if create_files:
        description = (
            "Execute Python code in a secure sandbox environment. Use this tool for:\n"
            "1. Complex mathematical computations (algebra, calculus, statistics, etc.)\n"
            "2. Data processing and analysis\n"
            "3. Creating files for constructing file-based questions\n\n"
            "Available modules and file types:\n"
            "• PDF files (.pdf): reportlab or fpdf\n"
            "• Parquet files (.parquet): pyarrow or fastparquet\n"
            "• HTML files (.html): Use open() from standard library\n"
            "• Excel files (.xlsx): openpyxl or pandas\n"
            "• JSON files (.json): json (standard library)\n"
            "• SQLite database files (.db, .sqlite): sqlite3 (standard library)\n"
            "• XML files (.xml): xml.etree.ElementTree (standard library)\n"
            "• YAML files (.yaml, .yml): PyYAML\n"
            "• Text files (.txt): Use open() from standard library\n"
            "• CSV files (.csv): csv (standard library) or pandas\n"
            "• HDF5 files (.h5): h5py or pandas\n"
            "• Pickle files (.pkl): pickle (standard library)\n"
            "• LaTeX files (.tex): Use open() from standard library\n"
            "• Log files (.log): logging (standard library)\n"
            "• TAR files (.tar): tarfile (standard library)\n"
            "• GZ files (.gz): gzip (standard library)\n"
            "• ZIP files (.zip): zipfile (standard library)\n"
            "• Feather files (.feather): pyarrow or feather-format\n"
            "• MSG files (.msg): extract-msg\n"
            "• VCard files (.vcf): vobject\n"
            "Available modules: numpy, pandas, sympy, math, json, csv, datetime, random, collections, itertools, openpyxl, pyarrow, fastparquet, h5py, Pillow, fpdf, reportlab, pyyaml, extract-msg, vobject, pydub, requests, PyGithub, feather-format, xml.etree.ElementTree, sqlite3, pickle, logging, tarfile, gzip, zipfile, wave\n"
            f"Use print() to output results. FILES_OUTPUT_DIR = '{FILES_OUTPUT_DIR}' variable is available for saving files\n"
            "Saved filenames must be randomly generated with 10 letters plus extension to ensure uniqueness. Example usage: filename = ''.join(random.choices(string.ascii_letters, k=10)) + '.parquet'"
        )
    else:
        description = (
            "Execute Python code in a secure sandbox environment. Use this tool for:\n"
            "1. Complex mathematical computations (algebra, calculus, statistics, etc.)\n"
            "2. Data processing and analysis\n"
            "Available modules: numpy, pandas, sympy, math, json, csv, datetime, random, collections, itertools, requests\n"
            f"Use print() to output results."
        )
    return {
        "type": "function",
        "function": {
            "name": "code_executor",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }



if __name__ == "__main__":

    async def _test():
        print("Testing code_executor...")

        # Simple computation
        result = await code_executor("print(2 + 2 * 3)")
        print("Test 1 - Simple computation:", result)

        # Unsafe code (should be blocked)
        result = await code_executor("import subprocess; subprocess.run(['ls'])")
        print("Test 2 - Unsafe code:", result)

    asyncio.run(_test())

