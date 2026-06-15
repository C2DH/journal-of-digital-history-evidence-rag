#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python source/0_prepare_inputs_and_convert_papers_cpu.py
python source/1_process_review_files_cpu_gguf.py
python source/2_build_vector_database_cpu.py
python source/3_evidence_rag_audit_cpu_gguf.py
python source/4_export_editor_outputs.py
echo "CPU pipeline finished. Check outputs/"
