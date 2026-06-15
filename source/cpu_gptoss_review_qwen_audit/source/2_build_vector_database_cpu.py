from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common import clean_text, ensure_dirs, env, p, write_json

PAPERS_TEXT_DIR = p("PAPERS_TEXT_DIR", "data/papers_text")
VECTOR_DB_DIR = p("VECTOR_DB_DIR", "data/vector_db")
OUTPUTS_DIR = p("OUTPUTS_DIR", "outputs")

SECTION_PATTERNS = [
    "abstract", "introduction", "background", "related work", "literature review", "method",
    "methods", "methodology", "approach", "experiments", "evaluation", "results", "discussion",
    "limitations", "conclusion", "references", "appendix"
]


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def looks_like_heading(line: str) -> bool:
    line = clean_text(line)
    if not line or len(line.split()) > 9:
        return False
    low = re.sub(r"^\d+(?:\.\d+)*\s+", "", line.lower()).strip(":")
    return low in SECTION_PATTERNS or (line.isupper() and len(line.split()) <= 7)


def split_sections(text: str) -> list[dict[str, str]]:
    sections = []
    title = "Front Matter"
    buf = []
    for line in text.splitlines():
        if looks_like_heading(line):
            body = clean_text("\n".join(buf))
            if body:
                sections.append({"section_title": title, "text": body})
            title = clean_text(re.sub(r"^\d+(?:\.\d+)*\s+", "", line))
            buf = []
        else:
            buf.append(line)
    body = clean_text("\n".join(buf))
    if body:
        sections.append({"section_title": title, "text": body})
    return sections or [{"section_title": "Full Paper", "text": text}]


def chunk_text(text: str, max_words: int = 260, overlap_words: int = 45) -> list[str]:
    words = re.findall(r"\S+", text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def safe_collection(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)[:200]


def main() -> None:
    ensure_dirs(VECTOR_DB_DIR, OUTPUTS_DIR)
    model_name = env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedder = SentenceTransformer(model_name, device="cpu")
    dim = embedder.get_sentence_embedding_dimension()
    client = QdrantClient(path=str(VECTOR_DB_DIR))

    manifest = []
    for paper_path in tqdm(sorted(PAPERS_TEXT_DIR.glob("*.txt")), desc="Indexing papers"):
        paper_id = paper_path.stem
        collection = safe_collection(paper_id)
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

        full_text = clean_text(paper_path.read_text(encoding="utf-8", errors="ignore"))
        chunks = []
        global_idx = 1
        for sec_idx, section in enumerate(split_sections(full_text), start=1):
            if section["section_title"].lower() == "references":
                continue
            for local_idx, ch in enumerate(chunk_text(section["text"]), start=1):
                chunks.append({
                    "chunk_id": f"{paper_id}__chunk_{global_idx:04d}",
                    "paper_id": paper_id,
                    "source_file": str(paper_path),
                    "section_index": sec_idx,
                    "section_title": section["section_title"],
                    "local_chunk_index": local_idx,
                    "word_count": word_count(ch),
                    "text": f"Paper ID: {paper_id}\nSection: {section['section_title']}\n\n{ch}",
                })
                global_idx += 1

        if chunks:
            vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
            points = [PointStruct(id=i + 1, vector=vectors[i].tolist(), payload=chunks[i]) for i in range(len(chunks))]
            for start in range(0, len(points), 64):
                client.upsert(collection_name=collection, points=points[start:start+64], wait=True)
        manifest.append({"paper_id": paper_id, "collection_name": collection, "chunk_count": len(chunks), "section_count": len({c["section_title"] for c in chunks})})

    write_json(OUTPUTS_DIR / "vector_manifest.json", manifest)
    print(f"Indexed {len(manifest)} papers. Manifest: {OUTPUTS_DIR / 'vector_manifest.json'}")


if __name__ == "__main__":
    main()
