from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
ZH_SOURCE_DIR = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "zh_enterprise"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "backend" / "data" / "benchmark_raw" / "format_coverage"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "zh_enterprise_parser_regression_manifest.json"

IMAGE_QUOTE = "数据处理者向境外提供数据，应当履行数据安全保护义务，采取技术措施和其他必要措施。"


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    source_dir = output_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    files = build_sources(source_dir)
    manifest = build_manifest(files, source_dir=source_dir, dataset_name=args.dataset_name)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"documents={len(manifest['documents'])} cases={len(manifest['cases'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Chinese parser-regression format coverage manifest from real enterprise/policy source text."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-name", default="format_coverage_zh_parser")
    return parser


def build_sources(source_dir: Path) -> dict[str, Path]:
    data_export_md = ZH_SOURCE_DIR / "data_export_security_assessment.md"
    network_review_md = ZH_SOURCE_DIR / "network_security_review_rules.md"
    internal_control_md = ZH_SOURCE_DIR / "central_enterprise_internal_control_guidance.md"

    files = {
        ".txt": source_dir / "data_export_security_assessment.txt",
        ".md": source_dir / "data_export_security_assessment.md",
        ".markdown": source_dir / "network_security_review_rules.markdown",
        ".html": source_dir / "algorithm_recommendation_rules.html",
        ".htm": source_dir / "data_cross_border_rules.htm",
        ".pdf": source_dir / "enterprise_data_accounting.pdf",
        ".docx": source_dir / "central_enterprise_internal_control_guidance.docx",
        ".csv": source_dir / "data_export_security_assessment_requirements.csv",
        ".png": source_dir / "data_export_security_assessment_ocr.png",
        ".jpg": source_dir / "data_export_security_assessment_ocr.jpg",
        ".jpeg": source_dir / "data_export_security_assessment_ocr.jpeg",
    }

    write_text_fixture(files[".txt"], data_export_md)
    shutil.copyfile(data_export_md, files[".md"])
    shutil.copyfile(network_review_md, files[".markdown"])
    shutil.copyfile(ZH_SOURCE_DIR / "raw" / "algorithm_recommendation_rules.html", files[".html"])
    shutil.copyfile(ZH_SOURCE_DIR / "raw" / "data_cross_border_rules.htm", files[".htm"])
    shutil.copyfile(ZH_SOURCE_DIR / "enterprise_data_accounting.pdf", files[".pdf"])
    write_docx_fixture(files[".docx"], internal_control_md)
    write_csv_fixture(files[".csv"])

    missing_images = [path for suffix, path in files.items() if suffix in {".png", ".jpg", ".jpeg"} and not path.exists()]
    if missing_images:
        missing = "\n".join(str(path) for path in missing_images)
        raise FileNotFoundError(
            "Image fixtures are missing. Run scripts/render_format_coverage_images.ps1 first:\n" + missing
        )
    return files


def write_text_fixture(target: Path, source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    target.write_text(text, encoding="utf-8")


def write_csv_fixture(target: Path) -> None:
    rows = [
        {
            "条款": "第四条",
            "事项": "重要数据出境",
            "原文摘录": "数据处理者向境外提供重要数据",
            "处理要求": "应当通过所在地省级网信部门向国家网信部门申报数据出境安全评估",
        },
        {
            "条款": "第四条",
            "事项": "大量个人信息出境",
            "原文摘录": "处理100万人以上个人信息的数据处理者向境外提供个人信息",
            "处理要求": "应当申报数据出境安全评估",
        },
        {
            "条款": "第六条",
            "事项": "申报材料",
            "原文摘录": "申报数据出境安全评估，应当提交申报书、数据出境风险自评估报告、法律文件等材料",
            "处理要求": "提交申报书、自评估报告、法律文件和安全评估工作需要的其他材料",
        },
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["条款", "事项", "原文摘录", "处理要求"])
        writer.writeheader()
        writer.writerows(rows)


def write_docx_fixture(target: Path, source: Path, *, max_paragraphs: int = 30) -> None:
    lines = [
        line.strip()
        for line in source.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        if line.strip()
    ][:max_paragraphs]
    if not lines:
        raise ValueError(f"No source text found for DOCX fixture: {source}")
    paragraphs = [line[2:].strip() if line.startswith("# ") else line for line in lines]
    document_xml = build_document_xml(paragraphs)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        package.writestr("_rels/.rels", ROOT_RELS_XML)
        package.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS_XML)
        package.writestr("word/document.xml", document_xml)


def build_document_xml(paragraphs: list[str]) -> str:
    body = "\n".join(
        f"<w:p><w:r><w:t>{html.escape(paragraph)}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\"/></w:sectPr></w:body>"
        "</w:document>"
    )


def build_manifest(files: dict[str, Path], *, source_dir: Path, dataset_name: str) -> dict[str, Any]:
    documents = []
    cases = []
    for suffix, path in sorted(files.items()):
        info = document_info_for_suffix(suffix)
        document_id = f"format_coverage_zh_{suffix.lstrip('.')}"
        title = f"format_coverage:{suffix.lstrip('.')}:中文企业文档解析样本"
        documents.append(
            {
                "id": document_id,
                "title": title,
                "path": path.relative_to(source_dir.parent).as_posix(),
                "description": info["description"],
                "status": "active",
                "acl": [{"principal_type": "public"}],
                "metadata": {
                    "benchmark": "format_coverage",
                    "benchmark_role": "parser_regression",
                    "language": "zh",
                    "upload_suffix": suffix,
                    "source_derivation": info["source_derivation"],
                    "source_url": info["source_url"],
                    "source_org": info["source_org"],
                    "file_sha256": sha256_path(path),
                    "expected_parser": info["expected_parser"],
                    "not_main_effect_score": True,
                },
            }
        )
        cases.append(
            {
                "case_name": f"format_coverage_zh:{suffix.lstrip('.')}",
                "acting_user_email": "viewer@local.test",
                "question": info["question"],
                "expected_document_ids": [document_id],
                "expected_outcome": "format_parse",
                "expected_key_facts": [{"label": info["expected_fact"], "aliases": [info["expected_fact"]], "weight": 1.0}],
                "expected_evidence_markers": [],
                "scoring_notes": "Format coverage parser-regression case. It verifies upload, parser/chunking, ACL, and retrieval wiring for this suffix; it is not part of verified234 effect scoring.",
                "metadata": {
                    "benchmark": "format_coverage",
                    "upload_suffix": suffix,
                    "source_derivation": info["source_derivation"],
                    "not_main_effect_score": True,
                },
            }
        )
    return {
        "dataset_name": dataset_name,
        "schema_version": "format-coverage-parser-regression-v1",
        "description": (
            "Chinese parser-regression format coverage set. Original and derived fixtures come from public Chinese "
            "enterprise/policy source text and are intentionally separated from the verified234 RAG effect benchmark."
        ),
        "documents": documents,
        "cases": cases,
    }


def document_info_for_suffix(suffix: str) -> dict[str, str]:
    base = {
        ".txt": {
            "source_derivation": "derived_plain_text_from_official_markdown",
            "source_url": "https://www.cac.gov.cn/2022-07/07/c_1658811536396503.htm",
            "source_org": "国家互联网信息办公室",
            "expected_parser": "txt",
            "expected_fact": "申报数据出境安全评估，应当提交以下材料",
            "question": "TXT 解析样本中，申报数据出境安全评估需要提交哪些材料？",
            "description": "Plain-text fixture derived from the official Chinese data-export security assessment source.",
        },
        ".md": {
            "source_derivation": "cleaned_markdown_from_official_html",
            "source_url": "https://www.cac.gov.cn/2022-07/07/c_1658811536396503.htm",
            "source_org": "国家互联网信息办公室",
            "expected_parser": "markdown",
            "expected_fact": "数据出境风险自评估应重点评估出境数据的规模、范围、种类、敏感程度",
            "question": "Markdown 解析样本中，数据出境风险自评估要重点评估哪些数据因素？",
            "description": "Markdown fixture from cleaned official Chinese policy text.",
        },
        ".markdown": {
            "source_derivation": "renamed_markdown_from_official_html",
            "source_url": "https://www.cac.gov.cn/2022-01/04/c_1642894602182845.htm",
            "source_org": "国家互联网信息办公室等",
            "expected_parser": "markdown",
            "expected_fact": "掌握超过100万用户个人信息的网络平台运营者赴国外上市，必须申报网络安全审查",
            "question": "Markdown 后缀解析样本中，掌握超过多少用户个人信息的平台赴国外上市要申报网络安全审查？",
            "description": "Markdown-extension fixture derived from the official cybersecurity review policy text.",
        },
        ".html": {
            "source_derivation": "original_downloaded_official_html",
            "source_url": "https://www.miit.gov.cn/jgsj/waj/wjfb/art/2022/art_a6ae77ea1f5e401eb8cc6819b869fdfa.html",
            "source_org": "工业和信息化部等",
            "expected_parser": "html",
            "expected_fact": "算法推荐服务提供者应当坚持主流价值导向，优化算法推荐服务机制",
            "question": "HTML 原始网页样本中，算法推荐服务提供者应当坚持什么导向？",
            "description": "Original downloaded official HTML page for algorithm recommendation rules.",
        },
        ".htm": {
            "source_derivation": "original_downloaded_official_htm",
            "source_url": "https://www.cac.gov.cn/2024-03/22/c_1712776611775634.htm",
            "source_org": "国家互联网信息办公室",
            "expected_parser": "html",
            "expected_fact": "自由贸易试验区内数据处理者向境外提供负面清单外的数据，可以免予申报数据出境安全评估",
            "question": "HTM 原始网页样本中，自贸试验区数据处理者提供负面清单外数据可免予什么？",
            "description": "Original downloaded official HTM page for data cross-border flow rules.",
        },
        ".pdf": {
            "source_derivation": "original_downloaded_official_pdf",
            "source_url": "https://www.gov.cn/zhengce/zhengceku/2022-12/19/content_5732695.htm",
            "source_org": "财政部",
            "expected_parser": "pdf",
            "expected_fact": "企业数据资源相关会计处理暂行规定",
            "question": "PDF 解析样本是哪份企业数据资源会计处理文件？",
            "description": "Original downloaded official PDF for enterprise data-resource accounting.",
        },
        ".docx": {
            "source_derivation": "derived_docx_from_official_markdown",
            "source_url": "http://www.sasac.gov.cn/n2588035/n2588320/n2588335/c12670064/content.html",
            "source_org": "国务院国资委",
            "expected_parser": "docx",
            "expected_fact": "建立健全以风险管理为导向、合规管理监督为重点的内控体系",
            "question": "DOCX 解析样本中，中央企业内控体系建设强调以什么为导向？",
            "description": "DOCX parser fixture derived from official SASAC internal-control guidance text.",
        },
        ".csv": {
            "source_derivation": "derived_table_from_official_policy_articles",
            "source_url": "https://www.cac.gov.cn/2022-07/07/c_1658811536396503.htm",
            "source_org": "国家互联网信息办公室",
            "expected_parser": "csv",
            "expected_fact": "申报数据出境安全评估，应当提交申报书、数据出境风险自评估报告、法律文件等材料",
            "question": "CSV 解析样本中，数据出境安全评估申报材料包括哪些？",
            "description": "CSV parser fixture derived from official data-export security assessment articles.",
        },
        ".png": image_info(".png"),
        ".jpg": image_info(".jpg"),
        ".jpeg": image_info(".jpeg"),
    }
    return base[suffix]


def image_info(suffix: str) -> dict[str, str]:
    return {
        "source_derivation": "rendered_image_from_official_policy_quote",
        "source_url": "https://www.cac.gov.cn/2024-03/22/c_1712776611775634.htm",
        "source_org": "国家互联网信息办公室",
        "expected_parser": "image",
        "expected_fact": IMAGE_QUOTE,
        "question": f"{suffix.upper()} 图片 OCR 样本中，数据处理者向境外提供数据应履行什么义务？",
        "description": f"{suffix.upper()} image OCR fixture rendered from an official Chinese policy quote.",
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOCUMENT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""


if __name__ == "__main__":
    main()
