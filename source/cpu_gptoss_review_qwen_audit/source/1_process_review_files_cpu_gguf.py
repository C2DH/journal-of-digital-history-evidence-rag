from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

from common import clean_text, ensure_dirs, env, p, write_json
from gguf_llm import chat_json

REVIEW_ORGANIZED_DIR = p("REVIEW_ORGANIZED_DIR", "data/review_pdfs_organized")
PROCESSED_REVIEWS_DIR = p("PROCESSED_REVIEWS_DIR", "data/processed_reviews")

SYSTEM = """You extract structured peer-review comments from academic review text.
Return strict JSON only. Do not add markdown.
"""


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return clean_text("\n\n".join(page.extract_text() or "" for page in reader.pages))


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))


def fallback_split(review_text: str, review_id: str) -> list[dict]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n|(?m)^\s*(?:\d+[.)]|[-*])\s+", review_text) if b.strip()]
    comments = []
    for i, block in enumerate(blocks, start=1):
        if len(block.split()) < 5:
            continue
        comments.append({
            "review_id": review_id,
            "comment_id": f"{review_id}_comment_{i:03d}",
            "reviewer": "unknown",
            "round": "unknown",
            "comment_text": clean_text(block),
        })
    return comments


def process_one(path: Path) -> dict:
    review_id = path.stem
    review_text = pdf_to_text(path)
    user = f"""
Review file name: {path.name}

Extract all reviewer comments as JSON using this schema:
{{
  "review_id": "{review_id}",
  "reviewer": "Reviewer label if present, otherwise unknown",
  "round": "Review round if present, otherwise unknown",
  "comments": [
    {{
      "comment_id": "{review_id}_comment_001",
      "comment_text": "one complete reviewer comment",
      "comment_type": "major|minor|summary|suggestion|question|other"
    }}
  ]
}}

Rules:
- Keep each comment complete and understandable.
- Do not invent content.
- Preserve criticism, praise, and suggestions.
- Return JSON only.

Review text:
{review_text[:24000]}
"""
    try:
        raw = chat_json(SYSTEM, user, max_tokens=int(env("MAX_TOKENS_REVIEW", "4096")), role="review")
        data = extract_json(raw)
        comments = data.get("comments", [])
        normalized = []
        for i, item in enumerate(comments, start=1):
            normalized.append({
                "review_id": review_id,
                "source_file": str(path),
                "comment_id": item.get("comment_id") or f"{review_id}_comment_{i:03d}",
                "reviewer": data.get("reviewer") or item.get("reviewer") or "unknown",
                "round": data.get("round") or item.get("round") or "unknown",
                "comment_type": item.get("comment_type", "other"),
                "comment_text": clean_text(item.get("comment_text", "")),
            })
        normalized = [c for c in normalized if c["comment_text"]]
        if not normalized:
            raise ValueError("Model returned no comments")
        return {"review_id": review_id, "comments": normalized, "raw_model_output": raw}
    except Exception as e:
        comments = fallback_split(review_text, review_id)
        return {"review_id": review_id, "comments": comments, "fallback_reason": str(e)}


def main() -> None:
    ensure_dirs(PROCESSED_REVIEWS_DIR)
    pdfs = sorted(REVIEW_ORGANIZED_DIR.glob("*.pdf"))
    all_comments = []
    reports = []
    for pdf in tqdm(pdfs, desc="Processing review PDFs"):
        result = process_one(pdf)
        reports.append(result)
        all_comments.extend(result["comments"])

    write_json(PROCESSED_REVIEWS_DIR / "review_processing_report.json", reports)
    write_json(PROCESSED_REVIEWS_DIR / "reviewer_comments.json", all_comments)
    print(f"Extracted {len(all_comments)} comments from {len(pdfs)} review PDFs.")
    print(f"Output: {PROCESSED_REVIEWS_DIR / 'reviewer_comments.json'}")


if __name__ == "__main__":
    main()
