from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "format_coverage"
DEFAULT_OUTPUT = DEFAULT_SOURCE_DIR / "manifest.json"


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_path = Path(args.output).resolve()
    manifest = build_manifest(source_dir, dataset_name=args.dataset_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"documents={len(manifest['documents'])} cases={len(manifest['cases'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a manifest that covers every supported upload suffix with real downloaded files."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dataset-name", default="format_coverage_real")
    return parser


def build_manifest(source_dir: Path, *, dataset_name: str) -> dict[str, Any]:
    sampled_csv = ensure_sampled_csv(source_dir / "owid-co2-data.csv")
    selected = {
        "txt": source_dir / "rfc9110.txt",
        "md": source_dir / "fastapi_docs" / "index.md",
        "markdown": source_dir / "fastapi_docs" / "first-steps.markdown",
        "html": pick_first(source_dir / "python_docs" / "html", "*.html", preferred_name="library/asyncio.html"),
        "htm": source_dir / "rfc9110.htm",
        "pdf": source_dir / "nist" / "NIST.SP.800-53r5.pdf",
        "docx": source_dir / "nist" / "FedRAMP-Security-Assessment-Report-SAR-Template.docx",
        "csv": sampled_csv,
        "png": pick_first(source_dir / "funsd" / "dataset" / "training_data" / "images", "*.png"),
        "jpg": source_dir / "sroie" / "ex.jpg",
        "jpeg": source_dir / "sroie" / "ex.jpeg",
    }
    missing = [f".{suffix}: {path}" for suffix, path in selected.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source files for format coverage:\n" + "\n".join(missing))

    metadata_by_suffix = {
        "txt": {
            "source_dataset": "RFC 9110",
            "source_url": "https://www.rfc-editor.org/rfc/rfc9110.txt",
            "expected_fact": "HTTP semantics include requests, responses, methods, status codes, and header fields.",
            "query": "Which HTTP specification document discusses methods, status codes, and header fields?",
        },
        "md": {
            "source_dataset": "FastAPI docs",
            "source_url": "https://github.com/fastapi/fastapi/blob/master/docs/en/docs/index.md",
            "expected_fact": "FastAPI is a modern, fast web framework for building APIs with Python.",
            "query": "Which imported Markdown document describes FastAPI as a modern web framework for building APIs with Python?",
        },
        "markdown": {
            "source_dataset": "FastAPI docs",
            "source_url": "https://github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/first-steps.md",
            "expected_fact": "The first steps tutorial shows importing FastAPI and creating an app instance.",
            "query": "Which Markdown tutorial shows importing FastAPI and creating an app instance?",
        },
        "html": {
            "source_dataset": "Python documentation",
            "source_url": "https://docs.python.org/3.12/archives/python-3.12-docs-html.zip",
            "expected_fact": "The Python documentation page covers asyncio APIs and asynchronous programming.",
            "query": "Which HTML document covers Python asyncio APIs and asynchronous programming?",
        },
        "htm": {
            "source_dataset": "RFC 9110 HTML",
            "source_url": "https://www.rfc-editor.org/rfc/rfc9110.html",
            "expected_fact": "RFC 9110 defines HTTP semantics.",
            "query": "Which HTM document defines HTTP semantics?",
        },
        "pdf": {
            "source_dataset": "NIST SP 800-53 Rev. 5",
            "source_url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
            "expected_fact": "NIST SP 800-53 provides security and privacy controls for information systems and organizations.",
            "query": "Which PDF provides security and privacy controls for information systems and organizations?",
        },
        "docx": {
            "source_dataset": "FedRAMP templates",
            "source_url": "https://www.fedramp.gov/resources/templates/FedRAMP-Security-Assessment-Report-(SAR)-Template.docx",
            "expected_fact": "The FedRAMP Security Assessment Report template documents security assessment results.",
            "query": "Which DOCX template is used to document FedRAMP security assessment results?",
        },
        "csv": {
            "source_dataset": "Our World in Data CO2",
            "source_url": "https://github.com/owid/co2-data/blob/master/owid-co2-data.csv",
            "expected_fact": "The OWID CO2 dataset contains country, year, iso_code, population, and emissions columns.",
            "query": "Which CSV dataset contains country, year, iso_code, population, and emissions columns?",
        },
        "png": {
            "source_dataset": "FUNSD",
            "source_url": "https://guillaumejaume.github.io/FUNSD/",
            "expected_fact": "FUNSD contains noisy scanned form images for form understanding.",
            "query": "Which PNG document comes from the FUNSD noisy scanned forms dataset?",
            "expected_outcome": "format_parse",
        },
        "jpg": {
            "source_dataset": "ICDAR 2019 SROIE sample",
            "source_url": "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE",
            "expected_fact": "SROIE contains scanned receipt images for OCR and key information extraction.",
            "query": "Which JPG document comes from the SROIE receipt OCR dataset?",
            "expected_outcome": "format_parse",
        },
        "jpeg": {
            "source_dataset": "ICDAR 2019 SROIE sample",
            "source_url": "https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE",
            "expected_fact": "SROIE contains scanned receipt images for OCR and key information extraction.",
            "query": "Which JPEG document comes from the SROIE receipt OCR dataset?",
            "expected_outcome": "format_parse",
        },
    }

    documents = []
    cases = []
    for suffix, path in selected.items():
        doc_id = f"format-{suffix}"
        title = f"format_coverage:{suffix}:{path.stem}"
        metadata = metadata_by_suffix[suffix]
        documents.append(
            {
                "id": doc_id,
                "title": title,
                "path": relative_to_source(source_dir, path),
                "description": f"Real {suffix.upper()} format coverage file from {metadata['source_dataset']}.",
                "status": "active",
                "acl": [{"principal_type": "public"}],
                "metadata": {
                    "benchmark": "format_coverage",
                    "upload_suffix": f".{suffix}",
                    "source_dataset": metadata["source_dataset"],
                    "source_url": metadata["source_url"],
                },
            }
        )
        cases.append(
            {
                "case_name": f"format_coverage:{suffix}",
                "acting_user_email": "viewer@local.test",
                "question": metadata["query"],
                "expected_document_ids": [doc_id],
                "expected_outcome": metadata.get("expected_outcome", "answer"),
                "expected_key_facts": [
                    {
                        "label": metadata["expected_fact"],
                        "aliases": [metadata["expected_fact"]],
                        "weight": 1.0,
                    }
                ],
                "scoring_notes": "Format coverage case: verifies upload suffix acceptance, parser output, chunking, ACL, and retrieval wiring for a real public file.",
                "metadata": {
                    "benchmark": "format_coverage",
                    "upload_suffix": f".{suffix}",
                    "source_dataset": metadata["source_dataset"],
                },
            }
        )

    return {
        "dataset_name": dataset_name,
        "documents": documents,
        "cases": cases,
    }


def pick_first(base_dir: Path, pattern: str, *, preferred_name: str | None = None) -> Path:
    if preferred_name:
        preferred = base_dir / preferred_name
        if preferred.exists():
            return preferred
        normalized_preferred = preferred_name.replace("\\", "/")
        for path in sorted(base_dir.rglob(pattern)):
            if path.is_file() and path.as_posix().endswith(normalized_preferred):
                return path
    for path in sorted(base_dir.rglob(pattern)):
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No files matched {pattern} under {base_dir}")


def ensure_sampled_csv(source_csv: Path, *, max_rows: int = 500) -> Path:
    if not source_csv.exists():
        raise FileNotFoundError(source_csv)
    target = source_csv.with_name(f"{source_csv.stem}.sample-{max_rows}{source_csv.suffix}")
    if target.exists() and target.stat().st_size > 0:
        return target

    preferred_countries = {"World", "United States", "China", "India", "Germany", "Brazil"}
    selected_rows: list[dict[str, str]] = []
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            country = row.get("country", "")
            year = row.get("year", "")
            if country in preferred_countries and year and int(year) >= 2000:
                selected_rows.append(row)
            if len(selected_rows) >= max_rows:
                break

    if not selected_rows:
        with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            selected_rows = [row for _, row in zip(range(max_rows), reader)]

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    return target


def relative_to_source(source_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(source_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
