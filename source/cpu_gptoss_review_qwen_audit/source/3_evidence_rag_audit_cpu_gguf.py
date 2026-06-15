from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common import clean_text, env, p, read_json, write_json
from gguf_llm import chat_json

PROCESSED_REVIEWS_DIR = p("PROCESSED_REVIEWS_DIR", "data/processed_reviews")
VECTOR_DB_DIR = p("VECTOR_DB_DIR", "data/vector_db")
OUTPUTS_DIR = p("OUTPUTS_DIR", "outputs")

SYSTEM = """You audit reviewer comments against paper evidence.
Return strict JSON only. Do not add markdown.
"""


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found")
    return json.loads(m.group(0))


def clamp01(x: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return default


def normalize_score(score: float) -> float:
    # Qdrant cosine scores are commonly 0..1 for normalized embeddings; keep safe.
    return clamp01(score)


def section_coverage(evidence: list[dict], total_sections: int) -> float:
    if total_sections <= 0:
        return 0.0
    return clamp01(len({e.get("section_title") for e in evidence if e.get("section_title")}) / total_sections)


def infer_paper_collection(comment: dict, manifest: list[dict]) -> dict:
    # If the comment contains paper_XXX or paper id, route to it. Otherwise use first paper.
    text = " ".join(str(comment.get(k, "")) for k in ["paper_id", "review_id", "source_file", "comment_id"])
    for item in manifest:
        if item["paper_id"] in text:
            return item
    return manifest[0]


def make_user_prompt(comment: dict, evidence: list[dict]) -> str:
    evidence_text = "\n\n".join(
        f"[Evidence {i}] chunk_id={e['chunk_id']} section={e.get('section_title')} retrieval_score={e.get('retrieval_score'):.4f}\n{e.get('text','')[:1800]}"
        for i, e in enumerate(evidence, start=1)
    )
    return f"""
Reviewer comment ID: {comment.get('comment_id')}
Reviewer comment:
{comment.get('comment_text')}

Retrieved paper evidence:
{evidence_text}

Return JSON using this schema:
{{
  "comment_id": "{comment.get('comment_id')}",
  "support_label": "supported|partially_supported|not_supported|unclear",
  "model_support_strength": 0.0,
  "evidence_specificity": 0.0,
  "reasoning_summary": "brief explanation grounded in the evidence",
  "best_evidence_chunk_ids": ["chunk id"]
}}

Rules:
- Judge only from the provided evidence.
- Do not invent paper content.
- model_support_strength: how strongly the evidence supports the reviewer comment.
- evidence_specificity: how directly the retrieved evidence addresses the exact reviewer claim.
- Return JSON only.
"""


def main() -> None:
    comments = read_json(PROCESSED_REVIEWS_DIR / "reviewer_comments.json", [])
    manifest = read_json(OUTPUTS_DIR / "vector_manifest.json", [])
    if not comments:
        raise RuntimeError("No reviewer comments found. Run file 1 first.")
    if not manifest:
        raise RuntimeError("No vector manifest found. Run file 2 first.")

    embedder = SentenceTransformer(env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"), device="cpu")
    client = QdrantClient(path=str(VECTOR_DB_DIR))
    results = []

    for comment in tqdm(comments, desc="Auditing comments"):
        paper_info = infer_paper_collection(comment, manifest)
        collection = paper_info["collection_name"]
        query_vec = embedder.encode(comment["comment_text"], normalize_embeddings=True).tolist()
        hits = client.query_points(collection_name=collection, query=query_vec, limit=6).points
        evidence = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload["retrieval_score"] = float(hit.score)
            evidence.append(payload)

        retrieval_strength = float(np.mean([normalize_score(e["retrieval_score"]) for e in evidence])) if evidence else 0.0
        total_sections = int(paper_info.get("section_count") or max(1, len({e.get("section_title") for e in evidence})))
        coverage = section_coverage(evidence, total_sections)

        try:
            raw = chat_json(SYSTEM, make_user_prompt(comment, evidence), max_tokens=int(env("MAX_TOKENS_AUDIT", "2048")), role="audit")
            judge = extract_json(raw)
        except Exception as e:
            raw = ""
            judge = {
                "comment_id": comment.get("comment_id"),
                "support_label": "unclear",
                "model_support_strength": 0.0,
                "evidence_specificity": 0.0,
                "reasoning_summary": f"Model JSON parsing failed: {e}",
                "best_evidence_chunk_ids": [],
            }

        model_support_strength = clamp01(judge.get("model_support_strength"))
        evidence_specificity = clamp01(judge.get("evidence_specificity"))
        confidence = (
            0.15 * retrieval_strength
            + 0.25 * model_support_strength
            + 0.45 * evidence_specificity
            + 0.15 * coverage
        )

        results.append({
            "comment_id": comment.get("comment_id"),
            "review_id": comment.get("review_id"),
            "paper_id": paper_info["paper_id"],
            "comment_text": comment.get("comment_text"),
            "support_label": judge.get("support_label", "unclear"),
            "reasoning_summary": judge.get("reasoning_summary", ""),
            "retrieval_strength": round(retrieval_strength, 4),
            "model_support_strength": round(model_support_strength, 4),
            "evidence_specificity": round(evidence_specificity, 4),
            "section_coverage": round(coverage, 4),
            "final_confidence": round(confidence, 4),
            "best_evidence_chunk_ids": judge.get("best_evidence_chunk_ids", []),
            "retrieved_evidence": [
                {
                    "chunk_id": e.get("chunk_id"),
                    "section_title": e.get("section_title"),
                    "retrieval_score": round(float(e.get("retrieval_score", 0.0)), 4),
                    "text": e.get("text", ""),
                }
                for e in evidence
            ],
            "raw_model_output": raw,
        })

    write_json(OUTPUTS_DIR / "evidence_rag_audit_results.json", results)
    print(f"Audited {len(results)} comments.")
    print(f"Output: {OUTPUTS_DIR / 'evidence_rag_audit_results.json'}")


if __name__ == "__main__":
    main()
