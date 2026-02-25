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
from typing import Any, Dict
from contextlib import redirect_stdout
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

logger = logging.getLogger(__name__)

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

# Allowed imports for computation and data processing
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


async def code_executor(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a safe sandbox environment.
    
    This tool can be used for:
    1. Complex mathematical computations that cannot be done mentally
    2. Data processing and analysis
    
    Args:
        code: Python code to execute. Use print() to output results.
              
    Returns:
        Dictionary containing:
        - stdout: Captured print output
        - result: Return value (if any)
        - error: Error message (if execution failed)
    """
    result = {
        "stdout": "",
        "result": None,
        "error": None,
    }
    
    try:
        runtime = SafeRuntime()
        
        # Clean up code
        code = code.strip()
        
        # Capture stdout
        stdout_capture = io.StringIO()
        
        def _execute():
            with redirect_stdout(stdout_capture):
                runtime.exec_code(code)
        
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


def get_openai_function_code_executor() -> dict:
    """Return the OpenAI tool/function definition for code_executor."""
    description = (
        "Execute Python code in a secure sandbox environment. Use this tool for:\n"
        "1. Complex mathematical computations (algebra, calculus, statistics, etc.)\n"
        "2. Data processing and analysis\n"
        "Available modules: numpy, pandas, sympy, math, json, csv, datetime, random, collections, itertools, requests\n"
        "Use print() to output results."
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

