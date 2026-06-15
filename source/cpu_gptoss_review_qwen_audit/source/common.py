from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def p(name: str, default: str) -> Path:
    raw = env(name, default)
    path = Path(raw)
    return path if path.is_absolute() else project_root() / path


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_name(text: str, max_len: int = 140) -> str:
    text = re.sub(r"https?://", "", text.strip())
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")[:max_len]
    return text or "item"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def copy_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
