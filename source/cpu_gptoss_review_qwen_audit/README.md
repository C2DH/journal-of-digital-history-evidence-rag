# JDH Evidence-RAG CPU Project: GPT-OSS Review Extraction + Qwen Audit Inference

This is a CPU-only version of the JDH Evidence-RAG workflow. It keeps the model mapping from our work:

| Pipeline step | Purpose | Model |
|---|---|---|
| File 1 / Step 1 | Extract structured reviewer comments from review PDFs | `unsloth/gpt-oss-120b-GGUF` |
| File 2 / Step 2 | Build the paper vector database | `BAAI/bge-small-en-v1.5` by default for CPU embeddings |
| File 3 / Step 3 | Final Evidence-RAG support audit | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF` |

The project uses `llama-cpp-python` and forces CPU-only inference with `n_gpu_layers=0`.

> **Important:** CPU inference with GPT-OSS-120B GGUF is possible with quantized files, but it can be very slow and needs a lot of RAM. The default review model quantization is `Q2_K` to make CPU use more feasible. The audit model keeps our original Qwen reasoning-distilled GGUF setup.

---

## Folder Structure

```text
cpu_gptoss_review_qwen_audit/
├── README.md
├── MODEL_MAP.md
├── DELETE_PREVIOUS_PROJECT.md
├── .env.example
├── .gitignore
├── install_cpu.sh
├── requirements.txt
├── run_pipeline.sh
├── data/
│   ├── paper_links/
│   │   └── paper_links.txt
│   ├── review_pdfs_upload/
│   │   └── .gitkeep
│   ├── papers_raw/
│   │   └── .gitkeep
│   ├── papers_text/
│   │   └── .gitkeep
│   ├── review_pdfs_organized/
│   │   └── .gitkeep
│   ├── processed_reviews/
│   │   └── .gitkeep
│   └── vector_db/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
└── source/
    ├── 0_prepare_inputs_and_convert_papers_cpu.py
    ├── 1_process_review_files_cpu_gguf.py
    ├── 2_build_vector_database_cpu.py
    ├── 3_evidence_rag_audit_cpu_gguf.py
    ├── 4_export_editor_outputs.py
    ├── common.py
    └── gguf_llm.py
```

---

## Installation

### Linux / macOS / WSL

```bash
cd cpu_gptoss_review_qwen_audit
bash install_cpu.sh
source .venv/bin/activate
```

If `llama-cpp-python` fails to build, install build tools first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev
bash install_cpu.sh
```

### Windows note

For Windows, the easiest route is WSL Ubuntu. You can also use PowerShell, but `llama-cpp-python` may need Visual Studio Build Tools and CMake.

---

## Configuration

Copy the environment template:

```bash
cp .env.example .env
```

Default model settings:

```bash
# File 1 / Step 1: review extraction
REVIEW_GGUF_REPO_ID=unsloth/gpt-oss-120b-GGUF
REVIEW_GGUF_FILENAME=Q2_K/gpt-oss-120b-Q2_K-00001-of-00002.gguf
REVIEW_GGUF_LOCAL_PATH=

# File 3 / Step 3: final Evidence-RAG audit inference
AUDIT_GGUF_REPO_ID=Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF
AUDIT_GGUF_FILENAME=Qwen3.5-27B.Q4_K_M.gguf
AUDIT_GGUF_LOCAL_PATH=
```

Use `*_GGUF_LOCAL_PATH` when you already downloaded the GGUF files into `models/`.

Example:

```bash
REVIEW_GGUF_LOCAL_PATH=models/gpt-oss-120b-Q2_K-00001-of-00002.gguf
AUDIT_GGUF_LOCAL_PATH=models/Qwen3.5-27B.Q4_K_M.gguf
```

---

## Add Input Files

### 1. Reviewer PDFs

Put all reviewer PDF files here:

```text
data/review_pdfs_upload/
```

The program automatically copies and organizes them into:

```text
data/review_pdfs_organized/
```

### 2. Paper links

Open this file:

```text
data/paper_links/paper_links.txt
```

Add one paper link per line:

```text
https://raw.githubusercontent.com/USER/REPO/main/paper_001.ipynb
https://example.org/paper_002.pdf
https://example.org/paper_003.txt
```

Supported inputs:

- `.ipynb`
- `.pdf`
- `.txt`
- `.md`
- `.json`
- `.html`
- GitHub `blob` links, which are converted to raw links automatically

---

## Run the Pipeline

Run all files in order:

```bash
bash run_pipeline.sh
```

Or run them manually:

```bash
python source/0_prepare_inputs_and_convert_papers_cpu.py
python source/1_process_review_files_cpu_gguf.py
python source/2_build_vector_database_cpu.py
python source/3_evidence_rag_audit_cpu_gguf.py
python source/4_export_editor_outputs.py
```

---

## Outputs

The main outputs are created in `outputs/`:

```text
outputs/input_manifest.json
outputs/vector_manifest.json
outputs/evidence_rag_audit_results.json
outputs/editor_review_table.csv
outputs/editor_review_table.xlsx
```

The editor-facing table contains:

- reviewer comment
- support label
- reasoning summary
- retrieval strength
- model support strength
- evidence specificity
- section coverage
- final confidence score
- retrieved evidence
- empty editor decision fields

---

## Confidence Formula

```text
Confidence = 0.15 × Retrieval Strength
           + 0.25 × Model Support Strength
           + 0.45 × Evidence Specificity
           + 0.15 × Section Coverage
```

The editor remains the final human decision-maker. The system only provides an evidence layer for review auditing.
