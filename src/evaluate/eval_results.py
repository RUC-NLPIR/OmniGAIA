"""
eval_results.py — Re-evaluate existing agent results with LLM-based equivalence.

Reads a JSON file produced by ``run_base_agent.py`` (with or without
``--enable-active-perception``),
re-computes exact-match (EM) and LLM-based equivalence scores, and writes the
updated results + metrics back to disk.

Configuration file:
    config/config.json
"""
import os
import sys
import json
import asyncio
import logging
import httpx
import argparse
import re
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from config_loader import get_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# =============================================================================
# Configuration
# =============================================================================

MAX_RETRIES = 3
RETRY_DELAY_BASE = 2

# Evaluation LLM (loaded from config/config.json)
APP_CONFIG = get_config()
EVAL_MODEL_NAME = APP_CONFIG["evaluation"]["model"]

EVAL_ENDPOINTS = [
    {
        "base_url": APP_CONFIG["evaluation"]["base_url"],
        "api_key": APP_CONFIG["evaluation"]["api_key"],
        "model": EVAL_MODEL_NAME,
    },
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("EvalResults")

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
# Helper: Check Equivalence
# =============================================================================

def _get_eval_client(http_client: httpx.AsyncClient) -> Tuple[Any, str]:
    """Return an ``AsyncOpenAI`` client + model name for answer evaluation."""
    endpoint = random.choice(EVAL_ENDPOINTS)
    return (
        AsyncOpenAI(
            api_key=endpoint["api_key"],
            base_url=endpoint["base_url"],
            http_client=http_client,
        ),
        endpoint["model"],
    )


async def check_equivalence(
    client: Any, question: str, predicted: str, standard: str, model_name: Optional[str] = None
) -> Tuple[bool, str]:
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

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model_name or EVAL_MODEL_NAME,
                messages=messages,
                max_tokens=40960,
                extra_body={"chat_template_kwargs": {"thinking": False}},
                timeout=600
            )
            
            full_content = response.choices[0].message.content or ""
            content = full_content
            # Remove reasoning content if it exists (content after </think> tags)
            if '</think>' in content:
                content = content.split('</think>')[-1].strip()
            # Handle potential reasoning content if the eval model produces it
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
# Main Execution
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Re-run evaluation on existing results.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing previous results.")
    parser.add_argument("--test_file_path", type=str, default=None, help="Path to the test JSON file to recover missing category by matching question.")
    parser.add_argument('--concurrent_limit', type=int, default=64, help="Maximum number of concurrent API calls")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return
        
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except Exception as e:
        print(f"Failed to load input JSON: {e}")
        return
        
    if not isinstance(items, list):
        items = [items]

    # Optional: load test file and build question -> category map
    question_to_category = {}
    if args.test_file_path:
        if not os.path.exists(args.test_file_path):
            print(f"Test file not found: {args.test_file_path}")
            return
        try:
            with open(args.test_file_path, "r", encoding="utf-8") as f:
                test_items = json.load(f)
            if not isinstance(test_items, list):
                test_items = [test_items]
            for t in test_items:
                if isinstance(t, dict):
                    q = t.get("question")
                    c = t.get("category")
                    if q and c:
                        question_to_category[q] = c
        except Exception as e:
            print(f"Failed to load test JSON: {e}")
            return

    if AsyncOpenAI is None:
        raise ImportError("openai package is required for AsyncOpenAI client")
    eval_http_client = httpx.AsyncClient(timeout=3600, trust_env=False)
    clients = []

    semaphore = asyncio.Semaphore(args.concurrent_limit)
    
    print(f"Processing {len(items)} items with concurrency limit {args.concurrent_limit}...")

    async def process_item(client, item, semaphore):
        async with semaphore:
            question = item.get("question", "")
            predicted_answer = item.get("predicted_answer", "")
            if question and not item.get("category"):
                recovered = question_to_category.get(question)
                if recovered:
                    item["category"] = recovered
            
            # If predicted_answer is missing, try to recover from messages using the fallback logic
            if not predicted_answer:
                messages = item.get("messages", [])
                content = ""
                # Find last assistant message
                if messages and isinstance(messages, list):
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant":
                            raw_content = msg.get("content", "")
                            # Handle content being list or string
                            if isinstance(raw_content, list):
                                for part in raw_content:
                                    if isinstance(part, dict) and "text" in part:
                                        content += part["text"]
                                    elif isinstance(part, str):
                                        content += part
                            elif isinstance(raw_content, str):
                                content = raw_content
                            break
                
                if content:
                    matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    if matches:
                        predicted_answer = matches[-1].strip()
                    else:
                        # Fallback: Last 20 words
                        words = content.split()
                        predicted_answer = " ".join(words[-20:])

            ground_truth = item.get("answer", "")
            
            # Normalize and calculate EM
            def normalize_text(s):
                if not s: return ""
                s = str(s).lower()
                s = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                return " ".join(s.split())
                
            em_score = 1 if normalize_text(predicted_answer) == normalize_text(ground_truth) else 0
            
            # Prepare text for equivalence check
            eval_predicted_text = predicted_answer
            
            # Calculate LLM Equivalence
            if predicted_answer:
                if em_score == 1:
                    llm_equal = True
                    llm_eval_response = "EM is 1, skipping LLM evaluation."
                else:
                    ds_client, ds_model = _get_eval_client(eval_http_client)
                    llm_equal, llm_eval_response = await check_equivalence(
                        ds_client, question, eval_predicted_text, ground_truth, model_name=ds_model
                    )
            else:
                llm_equal = False
                llm_eval_response = "Predicted answer is empty, skipping LLM evaluation."
            llm_score = 1 if llm_equal else 0
            
            # Update item
            item["em_score"] = em_score
            item["llm_equal"] = llm_score
            item["llm_eval_response"] = llm_eval_response
            
            return item

    # Distribute tasks among clients
    tasks = [process_item(None, item, semaphore) for item in items]
    
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

    # Close clients
    for client in clients:
        await client.close()
    await eval_http_client.aclose()

    # Calculate metrics
    overall_metrics = calculate_metrics(results)
    
    # Calculate metrics by Level
    level_metrics = {}
    for level in ["Easy", "Medium", "Hard"]:
        level_results = [r for r in results if r.get("Level") == level]
        level_metrics[level] = calculate_metrics(level_results)
        
    # Calculate metrics by question category in fixed order
    category_metrics = calculate_category_metrics(results)

    # Output file handling
    # Construct new filename with updated metrics
    new_output_path = args.input_file
    
    avg_em = overall_metrics['em']
    avg_llm_equal = overall_metrics['llm_equal']
    
    # Update llmeq in filename
    if "llmeq" in new_output_path:
        new_output_path = re.sub(r'llmeq\d+\.\d+', f'llmeq{avg_llm_equal:.4f}', new_output_path)
    
    # Also update em in filename if present to keep it consistent
    if "_em" in new_output_path:
        new_output_path = re.sub(r'_em\d+\.\d+', f'_em{avg_em:.4f}', new_output_path)

    if new_output_path != args.input_file:
        print(f"Renaming output file from {os.path.basename(args.input_file)} to {os.path.basename(new_output_path)}")

    with open(new_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    # Remove old file if name changed
    if new_output_path != args.input_file and os.path.exists(args.input_file):
        try:
            os.remove(args.input_file)
            print(f"Removed old file: {args.input_file}")
        except OSError as e:
            print(f"Error removing old file: {e}")
        
    # Save metrics file
    base_name = os.path.splitext(new_output_path)[0]
    metrics_path = f"{base_name}_metrics.json"

    # Remove old metrics file if name changed
    if new_output_path != args.input_file:
        old_base_name = os.path.splitext(args.input_file)[0]
        old_metrics_path = f"{old_base_name}_metrics.json"
        if os.path.exists(old_metrics_path):
             try:
                os.remove(old_metrics_path)
                print(f"Removed old metrics file: {old_metrics_path}")
             except OSError:
                pass

    # Save Metrics JSON
    metrics_data = {
        "overall": overall_metrics,
        "by_level": level_metrics,
        "by_category": category_metrics,
    }
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print(f"Updated Results saved to {new_output_path}")
    print(f"Updated Metrics saved to {metrics_path}")
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

if __name__ == "__main__":
    asyncio.run(main())
