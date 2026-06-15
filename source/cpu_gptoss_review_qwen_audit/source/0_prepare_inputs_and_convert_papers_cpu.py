from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import nbformat
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tqdm import tqdm

from common import clean_text, copy_or_replace, ensure_dirs, env, p, safe_name, write_json

PAPER_LINKS_FILE = p("PAPER_LINKS_FILE", "data/paper_links/paper_links.txt")
REVIEW_UPLOAD_DIR = p("REVIEW_UPLOAD_DIR", "data/review_pdfs_upload")
PAPERS_RAW_DIR = p("PAPERS_RAW_DIR", "data/papers_raw")
PAPERS_TEXT_DIR = p("PAPERS_TEXT_DIR", "data/papers_text")
REVIEW_ORGANIZED_DIR = p("REVIEW_ORGANIZED_DIR", "data/review_pdfs_organized")
OUTPUTS_DIR = p("OUTPUTS_DIR", "outputs")


def normalize_github_url(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def extension_from_response(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".pdf", ".txt", ".md", ".ipynb", ".json", ".html", ".htm"}:
        return suffix
    if "pdf" in content_type:
        return ".pdf"
    if "json" in content_type:
        return ".json"
    if "html" in content_type:
        return ".html"
    return ".txt"


def download_link(url: str, idx: int) -> Path:
    clean_url = normalize_github_url(url)
    r = requests.get(clean_url, timeout=90, headers={"User-Agent": "jdh-evidence-rag-cpu"})
    r.raise_for_status()
    ext = extension_from_response(clean_url, r.headers.get("content-type", ""))
    digest = hashlib.sha1(clean_url.encode("utf-8")).hexdigest()[:10]
    name = f"paper_{idx:03d}_{safe_name(clean_url, 80)}_{digest}{ext}"
    out = PAPERS_RAW_DIR / name
    out.write_bytes(r.content)
    return out


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(f"\n\n[Page {i}]\n" + (page.extract_text() or ""))
    return clean_text("\n".join(pages))


def ipynb_to_text(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    parts = []
    for i, cell in enumerate(nb.cells, start=1):
        ctype = cell.get("cell_type", "")
        source = clean_text(cell.get("source", ""))
        if not source:
            continue
        parts.append(f"\n\n[{ctype.upper()} CELL {i}]\n{source}")
        if ctype == "code":
            outputs = []
            for out in cell.get("outputs", []):
                if "text" in out:
                    outputs.append(clean_text(out["text"]))
                elif "data" in out and "text/plain" in out["data"]:
                    outputs.append(clean_text(out["data"]["text/plain"]))
            if outputs:
                parts.append("\n[OUTPUT]\n" + clean_text("\n".join(outputs)))
    return clean_text("\n".join(parts))


def html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return clean_text(soup.get_text("\n"))


def json_to_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, dict):
        for key in ["paper_text", "text", "full_text", "content", "body"]:
            if key in data and data[key]:
                return clean_text(data[key])
    return clean_text(json.dumps(data, ensure_ascii=False, indent=2))


def convert_to_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_text(path)
    if suffix == ".ipynb":
        return ipynb_to_text(path)
    if suffix in {".html", ".htm"}:
        return html_to_text(path)
    if suffix == ".json":
        return json_to_text(path)
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))


def read_links() -> list[str]:
    if not PAPER_LINKS_FILE.exists():
        return []
    links = []
    for line in PAPER_LINKS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        links.append(line)
    return links


def organize_reviews() -> list[dict]:
    records = []
    for i, pdf in enumerate(sorted(REVIEW_UPLOAD_DIR.glob("*.pdf")), start=1):
        dst = REVIEW_ORGANIZED_DIR / f"review_{i:03d}_{safe_name(pdf.stem)}.pdf"
        copy_or_replace(pdf, dst)
        records.append({"review_id": f"review_{i:03d}", "original_file": str(pdf), "organized_file": str(dst)})
    return records


def main() -> None:
    ensure_dirs(REVIEW_UPLOAD_DIR, PAPERS_RAW_DIR, PAPERS_TEXT_DIR, REVIEW_ORGANIZED_DIR, OUTPUTS_DIR)
    links = read_links()
    paper_records = []

    for i, url in enumerate(tqdm(links, desc="Downloading papers"), start=1):
        raw_path = download_link(url, i)
        text = convert_to_text(raw_path)
        paper_id = f"paper_{i:03d}"
        text_path = PAPERS_TEXT_DIR / f"{paper_id}.txt"
        text_path.write_text(text, encoding="utf-8")
        paper_records.append({
            "paper_id": paper_id,
            "source_link": url,
            "raw_file": str(raw_path),
            "text_file": str(text_path),
            "word_count": len(re.findall(r"\S+", text)),
        })

    review_records = organize_reviews()
    write_json(OUTPUTS_DIR / "input_manifest.json", {"papers": paper_records, "reviews": review_records})
    print(f"Prepared {len(paper_records)} papers and {len(review_records)} review PDFs.")
    print(f"Manifest: {OUTPUTS_DIR / 'input_manifest.json'}")


if __name__ == "__main__":
    main()
