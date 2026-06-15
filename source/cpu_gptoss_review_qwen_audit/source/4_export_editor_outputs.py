from __future__ import annotations

import pandas as pd
from common import p, read_json

OUTPUTS_DIR = p("OUTPUTS_DIR", "outputs")


def main() -> None:
    results = read_json(OUTPUTS_DIR / "evidence_rag_audit_results.json", [])
    rows = []
    for r in results:
        evidence_lines = []
        for e in r.get("retrieved_evidence", []):
            evidence_lines.append(
                f"{e.get('chunk_id')} | {e.get('section_title')} | score={e.get('retrieval_score')}\n{e.get('text','')[:700]}"
            )
        rows.append({
            "paper_id": r.get("paper_id"),
            "review_id": r.get("review_id"),
            "comment_id": r.get("comment_id"),
            "reviewer_comment": r.get("comment_text"),
            "support_label": r.get("support_label"),
            "reasoning_summary": r.get("reasoning_summary"),
            "retrieval_strength": r.get("retrieval_strength"),
            "model_support_strength": r.get("model_support_strength"),
            "evidence_specificity": r.get("evidence_specificity"),
            "section_coverage": r.get("section_coverage"),
            "final_confidence": r.get("final_confidence"),
            "retrieved_evidence": "\n\n---\n\n".join(evidence_lines),
            "editor_decision_correct": "",
            "editor_name": "",
            "editor_notes": "",
        })

    df = pd.DataFrame(rows)
    csv_path = OUTPUTS_DIR / "editor_review_table.csv"
    xlsx_path = OUTPUTS_DIR / "editor_review_table.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print(f"CSV: {csv_path}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    main()
