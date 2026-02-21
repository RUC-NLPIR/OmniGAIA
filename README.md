<h1 style="text-align: center; font-size: 1.8em; margin-bottom: 0.75em;">
  <span style="color:#1628a7; font-weight:bold;">O</span><span style="color:#402b94; font-weight:bold;">m</span><span style="color:#673ea0; font-weight:bold;">n</span><span style="color:#8b16aa; font-weight:bold;">i</span>GAIA: Towards Native Omni-Modal AI Agents
</h1>

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=20&duration=3000&pause=1000&color=005DE3&center=true&vCenter=true&width=800&lines=OmniGAIA:+A+Benchmark+for+Omni-Modal+General+AI+Assistants;OmniAtlas:+A+Reasoning+Agent+with+Active+Perception" alt="Typing Animation" />
</div>

## 🎬 Demo

### Agentic Reasoning on "Image + Audio" Scenario

<div align="center">
  <video controls width="95%">
    <source src="./assets/demo_image_audio.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>


### Agentic Reasoning on "Video w/ Audio" Scenario

<div align="center">
  <video src="https://github.com/user-attachments/assets/df9f8619-6c45-462f-a809-5ffba2a82bd4" />
</div>


## 💡 Overview

**OmniGAIA** is a comprehensive benchmark designed to evaluate the capabilities of omni-modal general AI assistants. Unlike existing benchmarks that focus on a single modality, OmniGAIA requires agents to jointly reason over **video**, **audio**, and **image** inputs while leveraging external tools such as web search and code execution.

We also introduce **OmniAtlas**, an agentic reasoning system that extends a base LLM with *active perception* tools, enabling the model to request and examine additional media segments during multi-step reasoning.

### ✨ Key Highlights

- **Omni-Modal Benchmark:** 360 human-verified QA pairs spanning 9 domains, requiring joint understanding of video, audio, and image content.
- **Agentic Event-Graph Construction:** A novel pipeline that builds structured event graphs from multi-modal sources using Gemini-3-Flash and DeepSeek-V3.2 with tool-augmented reasoning.
- **External Tool Integration:** Agents are equipped with web search & browsing, code execution, and cross-modal retrieval tools.
- **OmniAtlas Agent:** A fine-tuned agent with active perception capabilities and preference learning via OmniDPO.
- **Multi-Dimensional Evaluation:** Tasks are categorised by difficulty (Easy / Medium / Hard) and domain, with both exact-match and LLM-based equivalence metrics.

### 📊 Benchmark Construction

<div align="center">
  <img src="./assets/omnigaia_construction.png" width="95%" />
</div>

The OmniGAIA construction pipeline consists of four stages:
1. **Data Collection** — Curating video (with audio) and image+audio sources from FineVideo, LongVideoBench, LongVideo-Reason, COCO 2017, and HuggingFace, covering 100+ diverse domains.
2. **Valuable Information Discovery** — Using Gemini-3-Flash to extract events, environmental analysis, audio analysis (ASR, speaker ID), and image understanding (OCR, objects, faces).
3. **Agentic Omni-Modal Event Graph Construction** — DeepSeek-V3.2 iteratively expands an initial event graph by planning next steps, acquiring new information via tools, and verifying factual correctness with LLM self-reflexion and human review.
4. **QA Generation & Quality Review** — Generating difficult, multi-hop QA pairs through event fuzzification, followed by LLM and human verification for correctness, task difficulty, answer uniqueness.

### 📈 Benchmark Statistics

<div align="center">
  <img src="./assets/omnigaia_statistics.png" width="95%" />
</div>

**Key numbers:**
- **360** QA pairs across **9** domains (Geography, History, Technology, Sports, Arts, Movies, Science, Finance, Food)
- **3** difficulty levels — Easy (33.9%), Medium (44.4%), Hard (21.7%)
- **Median video duration:** 242.2s | **Median audio duration:** 197.0s
- **99.7%** of tasks require visual perception; **99.7%** require audio perception
- **98.6%** require web search; **74.4%** require code / computation

### 🎯 Task Examples

<div align="center">
  <img src="./assets/omnigaia_examples.png" width="95%" />
</div>

### 🤖 OmniAtlas Training Pipeline

<div align="center">
  <img src="./assets/omniatlas_training.png" width="95%" />
</div>

OmniAtlas is trained in two stages:
1. **Trajectory Synthesis & Supervised Learning** — Gemini-3 provides step supervision while DeepSeek-V3.2 performs tool-augmented reasoning. Successful trajectories are used for SFT.
2. **OmniDPO: Fine-Grained Error Correction** — Gemini-3 identifies and corrects errors in failed trajectories across perception, reasoning, and tool-use dimensions, producing preference pairs for DPO training.


## 🔧 Installation

### Environment Setup

```bash
# Create conda environment
conda create -n omnigaia python=3.10
conda activate omnigaia

# Clone the repository
git clone https://github.com/anonymous/OmniGAIA-Anon.git
cd OmniGAIA-Anon

# Install dependencies
pip install -r requirements.txt
```

### System Dependencies

- **ffmpeg** is required for video/audio processing in OmniAtlas:
  ```bash
  # Ubuntu / Debian
  sudo apt-get install ffmpeg

  # macOS
  brew install ffmpeg

  # Windows (via Chocolatey)
  choco install ffmpeg
  ```

### Environment Variables

Create a `.env` file or export the following variables before running:

```bash
# ── Model Endpoints ──────────────────────────────────────────────
# Main agent model (for run_base_agent.py / run_omni_atlas.py)
# Pass via CLI args: --api_key, --api_base_url, --model_name

# ── Evaluation LLM ──────────────────────────────────────────────
export EVAL_BASE_URL="http://localhost:8089/v1"   # LLM endpoint for answer equivalence checking
export EVAL_API_KEY="your-eval-api-key"
export EVAL_MODEL="deepseek-v3"

# ── Tool: Web Search & Browsing ─────────────────────────────────
export SERPER_API_KEY="your-serper-api-key"        # Google Serper API (https://serper.dev)
export JINA_API_KEY="your-jina-api-key"            # Jina Reader API (https://jina.ai)

# ── Data & Cache Directories ────────────────────────────────────
export OMNIGAIA_DATA_DIR="./data"                  # Root directory for benchmark media files
export IMAGE_SAVE_DIR="./cache/searched_images"    # Downloaded web images
export WEB_CACHE_DIR="./cache"                     # Web request cache
export CODE_FILES_OUTPUT_DIR="./outputs/code_files" # Code executor output
```



## 🏃 Quick Start

### Pre-preparation

#### 1. Model Serving

Before running agents, ensure your LLM and auxiliary models are served via an OpenAI-compatible API (e.g. using [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), or a cloud API):

```bash
# Example: serve a Qwen model with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Omni-30B \
    --port 8000 \
    --trust-remote-code
```

#### 2. Benchmark Data

Place the benchmark JSON and media files under the `data/` directory:

```
data/
├── omnigaia_test.json      # Benchmark questions
├── video/                  # Video files referenced in questions
├── audio/                  # Audio files referenced in questions
└── image/                  # Image files referenced in questions
```

### Running the Baseline Agent

The baseline agent supports both **Gemini** and **Qwen** model families. The model family is auto-detected from the `--model_name` argument.

```bash
# ── Run with Gemini ──────────────────────────────────────────────
python src/run_base_agent.py \
    --input_file ./data/omnigaia_test.json \
    --api_key "YOUR_API_KEY" \
    --api_base_url "https://your-gemini-endpoint/v1" \
    --model_name "gemini-3-flash" \
    --concurrent_limit 5

# ── Run with Qwen (OpenAI-compatible endpoint) ──────────────────
python src/run_base_agent.py \
    --input_file ./data/omnigaia_test.json \
    --api_key "empty" \
    --api_base_url "http://localhost:8000/v1" \
    --model_name "qwen3-omni-30b" \
    --concurrent_limit 5
```

**Parameters:**

| Parameter | Description |
|---|---|
| `--input_file` | Path to the benchmark JSON file |
| `--api_key` | API key for the model endpoint |
| `--api_base_url` | Base URL of the model API |
| `--model_name` | Model identifier (auto-selects Gemini vs Qwen agent) |
| `--level` | Filter by difficulty: `Easy`, `Medium`, or `Hard` |
| `--max_items` | Limit the number of items to process |
| `--concurrent_limit` | Maximum concurrent API calls (default: 5) |
| `--use_asr` | Use Whisper ASR to convert audio to text (for text-only models) |

### Running the OmniAtlas Agent

OmniAtlas extends the base agent with active perception tools that allow the model to request specific video/audio/image segments during reasoning:

```bash
python src/run_omni_atlas.py \
    --input_file ./data/omnigaia_test.json \
    --api_key "empty" \
    --api_base_url "http://localhost:8000/v1" \
    --model_name "omniatlas" \
    --output_dir ./outputs \
    --concurrent_limit 5 \
    --request_timeout 600 \
    --item_timeout 1800
```

**Additional OmniAtlas Parameters:**

| Parameter | Description |
|---|---|
| `--output_dir` | Directory for results (default: `./outputs`) |
| `--request_timeout` | Per-request timeout in seconds (default: 600) |
| `--forced_final_timeout` | Timeout for forced final answer after max turns (default: 300) |
| `--ffmpeg_timeout` | Timeout for ffmpeg subprocess (default: 180) |
| `--item_timeout` | Max total processing time per item (default: 1800) |
| `--eval_timeout` | Timeout for LLM equivalence evaluation (default: 120) |
| `--skip_eval` | Skip LLM-based equivalence evaluation |



## 📊 Evaluation

### Automatic Evaluation

Both `run_base_agent.py` and `run_omni_atlas.py` automatically evaluate results after generation. The evaluation includes:

- **Exact Match (EM):** Normalised string comparison between the predicted answer and ground truth.
- **LLM Equivalence:** An LLM judge (e.g. DeepSeek-V3) determines whether the predicted answer is semantically equivalent to the ground truth.

Results and metrics are saved to the `outputs/` directory.

### Re-evaluate Existing Results

To re-run evaluation on previously generated results (e.g. with a different evaluation model):

```bash
python src/evaluate/eval_results.py \
    --input_file ./outputs/run_omni_atlas_omniatlas/run_20260101_120000_em0.2500_llmeq0.4000.json \
    --test_file_path ./data/omnigaia_test.json \
    --concurrent_limit 64
```

**Parameters:**

| Parameter | Description |
|---|---|
| `--input_file` | Path to the results JSON from a previous run |
| `--test_file_path` | (Optional) Original test JSON to recover missing category labels |
| `--concurrent_limit` | Maximum concurrent evaluation API calls (default: 64) |

### Output Format

Each run produces two files:
- `run_<timestamp>_em<score>_llmeq<score>.json` — Per-item results with predictions, messages, and scores.
- `run_<timestamp>_em<score>_llmeq<score>_metrics.json` — Aggregated metrics (overall, by difficulty level, and by category).

Example metrics output:
```
==================================================
Total Items:            360
Average EM Score:       0.2500
Average LLM Equal Score:0.4000
Average Tool Calls:     6.50
Non-Empty Answer Ratio: 0.9800
--------------------
Easy     (n=122): EM=0.3500, LLM_Eq=0.5200
Medium   (n=160): EM=0.2300, LLM_Eq=0.3800
Hard     (n=78 ): EM=0.1400, LLM_Eq=0.2600
--------------------
Geo.  (n=69 ): EM=0.2800, LLM_Eq=0.4200
Tech. (n=49 ): EM=0.2600, LLM_Eq=0.4100
...
==================================================
```



## 🛠️ Tools

OmniGAIA agents are equipped with the following external tools:

| Tool | Description | Key Dependencies |
|---|---|---|
| **Web Search** | Google search via Serper API with result caching | `aiohttp`, Serper API |
| **Page Browser** | Fetch and extract webpage content via Jina Reader API | `aiohttp`, `beautifulsoup4`, Jina API |
| **Code Executor** | Sandboxed Python execution with common scientific libraries | Built-in (`exec`/`eval`) |
| **Active Perception** *(OmniAtlas only)* | `read_video`, `read_audio`, `read_image` — request specific media segments during reasoning | `opencv-python`, `pydub`, `ffmpeg` |



## 📄 License

This project is released under the [MIT License](LICENSE).

