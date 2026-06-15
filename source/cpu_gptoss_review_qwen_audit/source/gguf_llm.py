from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from llama_cpp import Llama

from common import env

ModelRole = Literal["review", "audit"]


def _load_llm(role: ModelRole) -> Llama:
    """Load one CPU-only GGUF model.

    This project intentionally uses two models:
    - role="review": GPT-OSS-120B GGUF for extracting structured review comments.
    - role="audit": Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF for
      the final Evidence-RAG support judgment.

    Both are forced to CPU with n_gpu_layers=0.
    """
    role_upper = role.upper()
    local_path = env(f"{role_upper}_GGUF_LOCAL_PATH", "").strip()
    n_ctx = int(env("N_CTX", "32768"))
    n_threads = int(env("N_THREADS", str(os.cpu_count() or 8)))
    n_batch = int(env("N_BATCH", "512"))

    common_kwargs = dict(
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=0,  # CPU-only
        verbose=False,
    )

    if local_path:
        model_path = Path(local_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"{role_upper}_GGUF_LOCAL_PATH does not exist: {model_path}")
        return Llama(model_path=str(model_path), **common_kwargs)

    if role == "review":
        repo_id = env("REVIEW_GGUF_REPO_ID", "unsloth/gpt-oss-120b-GGUF")
        filename = env("REVIEW_GGUF_FILENAME", "Q2_K/gpt-oss-120b-Q2_K-00001-of-00002.gguf")
    else:
        repo_id = env("AUDIT_GGUF_REPO_ID", "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF")
        filename = env("AUDIT_GGUF_FILENAME", "Qwen3.5-27B.Q4_K_M.gguf")

    return Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        **common_kwargs,
    )


@lru_cache(maxsize=2)
def get_llm(role: ModelRole = "audit") -> Llama:
    return _load_llm(role)


def chat_json(
    system: str,
    user: str,
    max_tokens: int,
    temperature: float | None = None,
    role: ModelRole = "audit",
) -> str:
    llm = get_llm(role)
    temp = float(env("TEMPERATURE", "0.1")) if temperature is None else temperature
    response: dict[str, Any] = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"]
