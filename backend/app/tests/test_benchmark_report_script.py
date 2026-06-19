from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_report_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "report_benchmark_results.py"
    spec = importlib.util.spec_from_file_location("report_benchmark_results", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_hardness_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "report_benchmark_hardness.py"
    spec = importlib.util.spec_from_file_location("report_benchmark_hardness", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_backlog_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "report_enterprise_source_backlog.py"
    spec = importlib.util.spec_from_file_location("report_enterprise_source_backlog", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_validator_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "validate_benchmark_manifest.py"
    spec = importlib.util.spec_from_file_location("validate_benchmark_manifest", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_candidates_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "validate_enterprise_source_candidates.py"
    spec = importlib.util.spec_from_file_location("validate_enterprise_source_candidates", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_downloader_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "download_enterprise_source_candidates.py"
    spec = importlib.util.spec_from_file_location("download_enterprise_source_candidates", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_file_quality_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "report_enterprise_source_file_quality.py"
    spec = importlib.util.spec_from_file_location("report_enterprise_source_file_quality", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_source_seed_manifest_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "build_enterprise_source_seed_manifest.py"
    spec = importlib.util.spec_from_file_location("build_enterprise_source_seed_manifest", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_review_promotion_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "report_enterprise_review_promotion.py"
    spec = importlib.util.spec_from_file_location("report_enterprise_review_promotion", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_ingestion_quality_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "report_zh_ingestion_quality.py"
    spec = importlib.util.spec_from_file_location("report_zh_ingestion_quality", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_format_coverage_status_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "report_format_coverage_status.py"
    spec = importlib.util.spec_from_file_location("report_format_coverage_status", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latest_retrieval_report_discovers_dataset_report(tmp_path: Path, monkeypatch) -> None:
    module = _load_report_module()
    backend_dir = tmp_path / "backend"
    output_dir = backend_dir / "data" / "eval_outputs"
    output_dir.mkdir(parents=True)
    report_path = output_dir / "stard-zh-law-docs-small-retrieval-local.json"
    report_path.write_text(
        json.dumps(
            {
                "dataset_name": "stard_zh_law_docs_small",
                "top_k": 10,
                "case_count": 20,
                "generated_at": "2026-06-01T16:56:05+00:00",
                "document_scope": {"title_prefix": "stard_zh_law_docs_small:", "document_count": 120},
                "summary": {"recall_at_k_avg": 0.6, "evidence_recall_at_k_avg": 0.4667},
                "failure_cases": [{"case_name": "stard:0"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "BACKEND_DIR", backend_dir)

    report = module.latest_retrieval_report("stard_zh_law_docs_small")

    assert report is not None
    assert report["path"] == str(report_path)
    assert report["case_count"] == 20
    assert report["document_scope"]["document_count"] == 120
    assert report["summary"]["recall_at_k_avg"] == 0.6
    assert report["summary"]["evidence_recall_at_k_avg"] == 0.4667
    assert report["failure_case_count"] == 1


def test_format_coverage_status_separates_parser_support_from_effect_manifest(tmp_path: Path) -> None:
    module = _load_format_coverage_status_module()
    benchmark_manifest = tmp_path / "verified.json"
    benchmark_manifest.write_text(
        json.dumps(
            {
                "dataset_name": "zh_enterprise_v1_seed",
                "documents": [
                    {"id": "pdf_doc", "path": "raw/a.pdf", "metadata": {"source_format": "pdf", "benchmark_role": "effect"}},
                    {"id": "html_doc", "path": "raw/b.html", "metadata": {"source_format": "html", "benchmark_role": "effect"}},
                ],
                "cases": [{"case_name": "case:1"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_report(
        benchmark_manifest_path=benchmark_manifest,
        format_manifest_path=tmp_path / "missing-format-manifest.json",
    )
    markdown = module.render_markdown(report)

    assert report["support_consistent"] is True
    assert ".docx" in report["supported_suffixes"]
    assert ".xlsx" in report["unsupported_not_counted"]
    assert report["main_effect_benchmark"]["source_format_counts"] == {"html": 1, "pdf": 1}
    assert report["format_coverage_manifest"]["exists"] is False
    assert "do not claim a separate all-format effect benchmark" in report["format_coverage_manifest"]["note"]
    assert "verified234 retrieval metrics cover" in markdown


def test_latest_retrieval_report_ignores_other_datasets(tmp_path: Path, monkeypatch) -> None:
    module = _load_report_module()
    backend_dir = tmp_path / "backend"
    output_dir = backend_dir / "data" / "eval_outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "financebench-retrieval.json").write_text(
        json.dumps({"dataset_name": "financebench_small", "summary": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "BACKEND_DIR", backend_dir)

    assert module.latest_retrieval_report("stard_zh_law_docs_small") is None


def test_benchmark_hardness_report_flags_saturated_small_enterprise_gate(tmp_path: Path) -> None:
    module = _load_hardness_module()
    manifest_path = tmp_path / "manifest.json"
    diagnostics_path = tmp_path / "diagnostics.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "zh_enterprise_test",
                "documents": [
                    {
                        "id": "data_policy",
                        "title": "zh_enterprise_test:data_security:数据安全制度",
                        "metadata": {"domain": "data_security", "source_org": "安全部"},
                    },
                    {
                        "id": "finance_policy",
                        "title": "zh_enterprise_test:finance:数据资产制度",
                        "metadata": {"domain": "finance", "source_org": "财务部"},
                    },
                ],
                "cases": [
                    {
                        "case_name": "data_policy:qa:1",
                        "question": "客户手机号导出要谁审批？",
                        "expected_document_ids": ["data_policy"],
                        "expected_outcome": "answer",
                        "expected_evidence_markers": [
                            {"label": "客户手机号数据导出应由信息安全负责人审批", "aliases": []}
                        ],
                        "metadata": {"source_id": "data_policy"},
                    },
                    {
                        "case_name": "data_policy:scenario:1",
                        "question": "海外 HR 系统同步员工资料时哪些手续可以免掉？",
                        "expected_document_ids": ["data_policy"],
                        "expected_outcome": "answer",
                        "expected_evidence_markers": [
                            {
                                "label": "跨境人力资源管理确需向境外提供员工个人信息的免予申报数据出境安全评估",
                                "aliases": [],
                            }
                        ],
                        "metadata": {"case_type": "low_overlap_enterprise_scenario"},
                    },
                    {
                        "case_name": "data_policy:permission:viewer_denied",
                        "question": "客户手机号导出要谁审批？",
                        "expected_document_ids": [],
                        "expected_outcome": "refuse",
                        "expected_evidence_markers": [],
                        "metadata": {"permission_variant": "viewer_denied"},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    diagnostics_path.write_text(
        json.dumps(
            {
                "dataset_name": "zh_enterprise_test",
                "case_count": 3,
                "document_scope": {"document_count": 2},
                "ablations": [
                    {
                        "ablation": {"name": "full_local"},
                        "summary": {
                            "total_cases": 3,
                            "pass_count": 3,
                            "pass_rate": 1.0,
                            "recall_at_k_avg": 1.0,
                            "evidence_recall_at_k_avg": 1.0,
                            "stage_metrics": {
                                "lexical": {"evidence_recall_avg": 1.0},
                                "final": {"evidence_recall_avg": 1.0},
                            },
                        },
                    },
                    {
                        "ablation": {"name": "lexical_only"},
                        "summary": {
                            "total_cases": 3,
                            "pass_count": 3,
                            "pass_rate": 1.0,
                            "recall_at_k_avg": 1.0,
                            "evidence_recall_at_k_avg": 1.0,
                        },
                    },
                    {
                        "ablation": {"name": "no_expansion"},
                        "summary": {
                            "total_cases": 3,
                            "pass_count": 3,
                            "pass_rate": 1.0,
                            "recall_at_k_avg": 1.0,
                            "evidence_recall_at_k_avg": 1.0,
                        },
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_report(
        manifest_path=manifest_path,
        diagnostics_paths=[diagnostics_path],
        low_overlap_threshold=0.35,
        saturation_pass_rate=0.98,
    )

    assert report["manifest"]["document_count"] == 2
    assert report["manifest"]["case_type_counts"]["low_overlap_enterprise_scenario"] == 1
    assert "small_document_pool" in report["manifest"]["risk_flags"]
    assert report["manifest"]["query_evidence_overlap"]["low_overlap_count"] >= 1
    assert "ablation_saturated" in report["diagnostics"][0]["risk_flags"]
    assert "lexical_only_matches_default" in report["diagnostics"][0]["risk_flags"]
    assert any(item["action"] == "Do not tune retrieval parameters from this gate" for item in report["recommendations"])


def test_benchmark_hardness_markdown_contains_core_sections(tmp_path: Path) -> None:
    module = _load_hardness_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "demo",
                "documents": [{"id": "doc", "title": "demo:doc", "metadata": {"domain": "policy"}}],
                "cases": [
                    {
                        "case_name": "demo:1",
                        "question": "审批要求",
                        "expected_document_ids": ["doc"],
                        "expected_evidence_markers": [{"label": "审批要求包括负责人和时限"}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_report(
        manifest_path=manifest_path,
        diagnostics_paths=[],
        low_overlap_threshold=0.35,
        saturation_pass_rate=0.98,
    )
    markdown = module.render_markdown(report)

    assert "# Benchmark Hardness Report: demo" in markdown
    assert "## Manifest" in markdown
    assert "## Recommendations" in markdown


def test_enterprise_source_backlog_report_checks_targets(tmp_path: Path) -> None:
    module = _load_source_backlog_module()
    backlog_path = tmp_path / "source_backlog.json"
    backlog_path.write_text(
        json.dumps(
            {
                "name": "enterprise_backlog",
                "version": "v1",
                "target": {"documents_min": 10, "cases_min": 20},
                "acceptance_rules": [f"rule {index}" for index in range(6)],
                "case_mix": {"low_overlap_enterprise_scenario_min_rate": 0.35},
                "ablation_gate": {"required_ablations": ["full_local", "lexical_only"]},
                "source_collections": [
                    {
                        "id": "listed_company_internal_systems",
                        "priority": "P0",
                        "target_documents": 12,
                        "target_cases": 24,
                        "source_type": "上市公司制度公告",
                        "domains": ["procurement", "internal_control"],
                        "seed_queries": ["site:static.cninfo.com.cn 采购管理制度", "site:static.cninfo.com.cn 内控制度"],
                    },
                    {
                        "id": "format_coverage_only",
                        "priority": "P2",
                        "target_documents": 3,
                        "target_cases": 0,
                        "source_type": "格式覆盖",
                        "domains": ["format_pdf"],
                        "seed_queries": [],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_report(backlog_path)

    assert report["summary"]["target_documents"] == 12
    assert report["summary"]["target_cases"] == 24
    assert report["summary"]["seed_query_count"] == 2
    assert report["checks"]["target_documents_met"] is True
    assert report["checks"]["target_cases_met"] is True
    assert report["risks"] == []


def test_manifest_validator_flags_current_gate_quality_gaps(tmp_path: Path) -> None:
    module = _load_manifest_validator_module()
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "policy.md").write_text("审批制度正文，包含足够的中文内容。", encoding="utf-8")
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_name": "zh_enterprise_small",
                "documents": [
                    {
                        "id": "policy",
                        "title": "zh_enterprise_small:policy:审批制度",
                        "path": "policy.md",
                        "acl": [{"principal_type": "public"}],
                        "metadata": {
                            "source_url": "https://example.com/policy.md",
                            "source_org": "示例公司",
                            "language": "zh",
                            "domain": "procurement",
                        },
                    }
                ],
                "cases": [
                    {
                        "case_name": "policy:qa:1",
                        "acting_user_email": "viewer@local.test",
                        "question": "采购审批要准备什么？",
                        "expected_document_ids": ["policy"],
                        "expected_outcome": "answer",
                        "expected_key_facts": [{"label": "采购审批需要提交预算、合同和供应商评估材料"}],
                        "expected_evidence_markers": [{"label": "采购审批需要提交预算、合同和供应商评估材料"}],
                        "metadata": {"case_type": "single_fact"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.validate_manifest(
        manifest_path=manifest_path,
        source_dir=source_dir,
        min_documents=2,
        min_cases=2,
        min_low_overlap_rate=0.35,
        min_multi_evidence_rate=0.25,
        min_cross_document_rate=0.10,
        min_permission_cases=1,
    )

    assert report["passed"] is False
    assert "document_count_below_v1_min:1<2" in report["errors"]
    assert "case_count_below_v1_min:1<2" in report["errors"]
    assert any(error.startswith("multi_evidence_rate_below_v1_min") for error in report["errors"])


def test_source_candidates_validator_accepts_screened_official_sources(tmp_path: Path) -> None:
    module = _load_source_candidates_module()
    candidates_path = tmp_path / "source_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "name": "candidate_test",
                "version": "2026-06-08",
                "candidate_sources": [
                    {
                        "candidate_id": "cninfo-policy",
                        "collection_id": "listed_company_internal_systems",
                        "title": "采购管理制度",
                        "source_org": "示例公司",
                        "source_url": "https://static.cninfo.com.cn/finalpage/2025-01-01/1.PDF",
                        "source_platform": "巨潮资讯网",
                        "doc_type": "procurement_policy",
                        "source_format": "pdf",
                        "retrieval_method": "direct_pdf",
                        "stability_status": "direct_download_url",
                        "benchmark_role": "effect",
                        "selection_status": "screened",
                        "expected_case_types": ["low_overlap_enterprise_scenario"],
                        "risk_notes": "下载后确认正文长度。",
                    },
                    {
                        "candidate_id": "csg-html",
                        "collection_id": "procurement_supplier_platforms",
                        "title": "供应商资格预审公告",
                        "source_org": "南方电网",
                        "source_url": "https://www.bidding.csg.cn/gywgg/1200356385.jhtml",
                        "source_platform": "南方电网供应链统一服务平台",
                        "doc_type": "supplier_prequalification_notice",
                        "source_format": "html",
                        "retrieval_method": "official_html",
                        "stability_status": "official_page",
                        "benchmark_role": "effect",
                        "selection_status": "screened",
                        "expected_case_types": ["supplier_qualification"],
                        "risk_notes": "需清洗 UI 噪声。",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = module.build_report(
        candidates_path,
        min_candidates=2,
        min_collections=2,
        min_direct_download=1,
    )

    assert report["passed"] is True
    assert report["summary"]["official_source_count"] == 2
    assert report["summary"]["direct_download_count"] == 1
    assert report["summary"]["ready_to_download_count"] == 1


def test_source_candidate_downloader_selects_only_downloadable_sources(tmp_path: Path) -> None:
    module = _load_source_downloader_module()
    direct_pdf = {
        "candidate_id": "cninfo-policy",
        "selection_status": "screened",
        "retrieval_method": "direct_pdf",
        "source_format": "pdf",
        "source_url": "https://static.cninfo.com.cn/finalpage/2025-01-01/1.PDF",
    }
    html_page = {
        "candidate_id": "csg-html",
        "selection_status": "screened",
        "retrieval_method": "official_html",
        "source_format": "html",
        "source_url": "https://www.bidding.csg.cn/gywgg/1200356385.jhtml",
    }
    attachment_page = {
        "candidate_id": "chinamoney-page",
        "selection_status": "screened",
        "retrieval_method": "official_html_then_attachment",
        "source_format": "html_with_pdf_attachment",
        "source_url": "https://www.chinamoney.com.cn/chinese/zqfxgg/20260409/3311795.html",
    }

    selected_default = module.select_candidates(
        [direct_pdf, html_page, attachment_page],
        limit=None,
        collection_id=None,
        include_html=False,
    )
    selected_with_html = module.select_candidates(
        [direct_pdf, html_page, attachment_page],
        limit=None,
        collection_id=None,
        include_html=True,
    )
    selected_with_attachments = module.select_candidates(
        [direct_pdf, html_page, attachment_page],
        limit=None,
        collection_id=None,
        include_html=False,
        include_attachments=True,
    )

    assert [item["candidate_id"] for item in selected_default] == ["cninfo-policy"]
    assert [item["candidate_id"] for item in selected_with_html] == ["cninfo-policy", "csg-html"]
    assert [item["candidate_id"] for item in selected_with_attachments] == ["cninfo-policy", "chinamoney-page"]
    assert module.skip_reason(attachment_page, include_html=True) == "requires_attachment_extractor"
    assert module.skip_reason(attachment_page, include_html=True, include_attachments=True) is None
    assert module.download_path_for_candidate(tmp_path, direct_pdf).name == "cninfo-policy.pdf"


def test_source_candidate_downloader_extracts_official_attachment_links() -> None:
    module = _load_source_downloader_module()
    shclearing_html = """
      <script>
        var fileNames = './P0201.pdf;;./P0202.pdf';
        var descNames = '发行方案.pdf;;北控水务集团有限公司2025年度第一期中期票据募集说明书.pdf';
      </script>
    """
    chinamoney_html = """
      <a href="javascript:void(0);" onclick="location.href=encodeURI($('#fileDownUrl').val()+'fileDownLoad.do?mode=open&contentId=3311795&priority=0');">
        <span>深圳市环境水务集团有限公司2026年度第一期中期票据募集说明书.pdf</span>
      </a>
    """

    sh_attachment = module.find_attachment(
        shclearing_html,
        "https://www.shclearing.com.cn/xxpl/fxpl/mtn/202501/t.html",
        {"title": "北控水务集团有限公司2025年度第一期中期票据发行文件", "doc_type": "mtn_prospectus_page"},
    )
    cm_attachment = module.find_attachment(
        chinamoney_html,
        "https://www.chinamoney.com.cn/chinese/zqfxgg/20260409/3311795.html?cp=zqfx",
        {"title": "深圳市环境水务集团有限公司2026年度第一期中期票据募集说明书", "doc_type": "mtn_prospectus_page"},
    )

    assert "P0202.pdf" in sh_attachment["url"]
    assert "募集说明书" in sh_attachment["name"]
    assert cm_attachment["url"].endswith("fileDownLoad.do?mode=open&contentId=3311795&priority=0")


def test_source_file_quality_classifies_text_thresholds() -> None:
    module = _load_source_file_quality_module()

    accepted = module.classify_text_quality(
        {"candidate_id": "long", "page_count": 12},
        "企" * 8000,
        min_long_cjk=8000,
        min_review_cjk=3000,
        min_review_pages=4,
    )
    review = module.classify_text_quality(
        {"candidate_id": "review", "page_count": 5},
        "企" * 3000,
        min_long_cjk=8000,
        min_review_cjk=3000,
        min_review_pages=4,
    )
    rejected = module.classify_text_quality(
        {"candidate_id": "short", "page_count": 2},
        "企" * 100,
        min_long_cjk=8000,
        min_review_cjk=3000,
        min_review_pages=4,
    )

    assert accepted["quality_status"] == "accepted_effect_long"
    assert review["quality_status"] == "needs_case_review_short_but_usable"
    assert rejected["quality_status"] == "reject_too_short"


def test_source_file_quality_ui_noise_ignores_business_register_terms() -> None:
    module = _load_source_file_quality_module()

    assert module.find_strict_ui_noise("公司注册资本为人民币一亿元，发行注册程序已经履行。") == []
    assert module.find_strict_ui_noise("首页 登录 注册 搜索") == ["登录/注册导航"]
    assert module.find_strict_ui_noise("打开微信扫一扫") == ["打开微信", "扫一扫"]


def test_source_file_quality_flags_invalid_pdf_even_with_matching_checksum(tmp_path: Path) -> None:
    module = _load_source_file_quality_module()
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("not a pdf", encoding="utf-8")
    record = {
        "candidate_id": "fake",
        "target_path": str(fake_pdf),
        "file_sha256": module.sha256_path(fake_pdf),
        "bytes": fake_pdf.stat().st_size,
    }

    quality = module.inspect_record(record, min_long_cjk=8000, min_review_cjk=3000, min_review_pages=4)

    assert quality["checksum_match"] is True
    assert quality["quality_status"] == "invalid_file_type"


def test_source_seed_manifest_uses_only_accepted_quality_rows(tmp_path: Path) -> None:
    module = _load_source_seed_manifest_module()
    candidates_path = tmp_path / "candidates.json"
    downloads_path = tmp_path / "downloads.json"
    quality_path = tmp_path / "quality.json"
    source_file = tmp_path / "raw" / "accepted.pdf"
    source_file.parent.mkdir()
    source_file.write_bytes(b"%PDF demo")
    candidates_path.write_text(
        json.dumps(
            {
                "candidate_sources": [
                    {
                        "candidate_id": "accepted-doc",
                        "source_org": "示例公司",
                        "benchmark_role": "effect",
                        "source_format": "pdf",
                    },
                    {
                        "candidate_id": "bad-doc",
                        "source_org": "示例公司",
                        "benchmark_role": "effect",
                        "source_format": "pdf",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    downloads_path.write_text(
        json.dumps(
            {
                "downloads": [
                    {"candidate_id": "accepted-doc", "status": "downloaded", "retrieved_at": "2026-06-08T00:00:00+00:00"},
                    {"candidate_id": "bad-doc", "status": "downloaded", "retrieved_at": "2026-06-08T00:00:00+00:00"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    quality_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "candidate_id": "accepted-doc",
                        "quality_status": "accepted_effect_long",
                        "target_path": str(source_file),
                        "title": "采购制度",
                        "source_url": "https://static.cninfo.com.cn/finalpage/1.PDF",
                        "source_platform": "巨潮资讯网",
                        "actual_sha256": "a" * 64,
                        "doc_type": "procurement_policy",
                        "domain": "procurement",
                        "page_count": 12,
                        "cjk_chars": 9000,
                    },
                    {
                        "candidate_id": "bad-doc",
                        "quality_status": "invalid_file_type",
                        "target_path": str(source_file),
                        "title": "坏文件",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = module.build_manifest(
        candidates_path=candidates_path,
        download_report_path=downloads_path,
        quality_report_path=quality_path,
        dataset_name="zh_enterprise_v1_seed_test",
        accept_statuses={"accepted_effect_long"},
        metadata_overrides_path=None,
    )

    assert len(manifest["documents"]) == 1
    assert manifest["documents"][0]["id"] == "accepted_doc"
    assert manifest["documents"][0]["metadata"]["file_sha256"] == "a" * 64
    assert manifest["cases"] == []


def test_source_seed_manifest_applies_metadata_overrides(tmp_path: Path) -> None:
    module = _load_source_seed_manifest_module()
    candidates_path = tmp_path / "candidates.json"
    downloads_path = tmp_path / "downloads.json"
    quality_path = tmp_path / "quality.json"
    overrides_path = tmp_path / "overrides.json"
    source_file = tmp_path / "source.html"
    source_file.write_text("正文", encoding="utf-8")

    candidates_path.write_text(
        json.dumps(
            {
                "candidate_sources": [
                    {
                        "candidate_id": "placeholder-doc",
                        "collection_id": "procurement_supplier_platforms",
                        "title": "截断标题 ...",
                        "source_org": "待下载后从文档首页确认",
                        "source_format": "html",
                        "benchmark_role": "effect",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    downloads_path.write_text(
        json.dumps(
            {
                "downloads": [
                    {
                        "candidate_id": "placeholder-doc",
                        "status": "downloaded",
                        "retrieved_at": "2026-06-08T00:00:00+00:00",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    quality_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "candidate_id": "placeholder-doc",
                        "quality_status": "needs_case_review_short_but_usable",
                        "target_path": str(source_file),
                        "title": "截断标题 ...",
                        "source_url": "https://www.bidding.csg.cn/zbgg/1.jhtml",
                        "source_platform": "南方电网供应链统一服务平台",
                        "doc_type": "tender_or_supplier_notice",
                        "domain": "procurement",
                        "cjk_chars": 6500,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    overrides_path.write_text(
        json.dumps(
            {
                "overrides": {
                    "placeholder-doc": {
                        "source_org": "广东电网有限责任公司",
                        "title": "广东电网公司真实招标公告",
                        "evidence": "HTML正文：招标人为广东电网有限责任公司。",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = module.build_manifest(
        candidates_path=candidates_path,
        download_report_path=downloads_path,
        quality_report_path=quality_path,
        dataset_name="zh_enterprise_v1_seed_test",
        accept_statuses={"needs_case_review_short_but_usable"},
        metadata_overrides_path=overrides_path,
    )

    document = manifest["documents"][0]
    assert document["title"] == "zh_enterprise_v1_seed_test:procurement:广东电网有限责任公司:广东电网公司真实招标公告"
    assert document["metadata"]["source_org"] == "广东电网有限责任公司"
    assert document["metadata"]["metadata_override_applied"] is True
    assert "招标人为广东电网有限责任公司" in document["metadata"]["source_org_evidence"]


def test_ingestion_quality_manifest_metadata_gap_detection() -> None:
    module = _load_ingestion_quality_module()

    manifest_doc = {
        "metadata": {
            "source_url": "https://example.com/policy.pdf",
            "source_org": "示例公司",
            "language": "zh",
            "file_sha256": "a" * 64,
            "retrieved_at": "2026-06-08T00:00:00+00:00",
            "doc_type": "policy",
            "domain": "procurement",
            "benchmark_role": "effect",
            "source_platform": "官方平台",
            "source_candidate_id": "sample",
            "quality_status": "accepted_effect_long",
        }
    }

    assert module.missing_manifest_metadata(manifest_doc) == ["source_format"]


def test_ingestion_quality_table_signal_and_summary_gate() -> None:
    module = _load_ingestion_quality_module()

    assert module.matched_noise_terms("供应商登录供应链统一服务平台获取电子招标文件") == []
    assert module.matched_noise_terms("首页 登录 注册 搜索") == ["登录/注册导航"]
    assert module.is_table_signal_text("Table row: 供应商 | 准入材料", chunk_type=None)
    assert module.is_table_signal_text("供应商字段=准入材料", chunk_type=None)
    assert module.is_table_signal_text("普通段落", chunk_type="table")
    assert not module.is_table_signal_text("普通段落", chunk_type="paragraph")

    summary = module.summarize_items(
        [
            {
                "found_in_db": True,
                "ingest_status": "ready",
                "chunk_count": 8,
                "table_signal_chunk_count": 2,
                "strict_noise_count": 0,
                "noisy_chunk_count": 0,
                "checksum_match": True,
                "manifest_metadata_missing": [],
                "lexical_missing_count": 0,
            },
            {
                "found_in_db": True,
                "ingest_status": "ready",
                "chunk_count": 4,
                "table_signal_chunk_count": 0,
                "strict_noise_count": 1,
                "noisy_chunk_count": 1,
                "checksum_match": False,
                "manifest_metadata_missing": ["source_format"],
                "lexical_missing_count": 1,
            },
            {"found_in_db": False},
        ],
        min_table_signal_doc_rate=0.5,
    )

    assert summary["found_document_count"] == 2
    assert summary["missing_document_count"] == 1
    assert summary["table_signal_doc_rate"] == 0.5
    assert summary["table_signal_gate_passed"] is True
    assert summary["checksum_mismatch_count"] == 1
    assert summary["manifest_metadata_gap_count"] == 1


def test_review_promotion_keeps_placeholder_and_promotes_clean_review(tmp_path: Path) -> None:
    module = _load_review_promotion_module()
    long_manifest_path = tmp_path / "long.json"
    review_manifest_path = tmp_path / "review.json"
    ingestion_report_path = tmp_path / "ingestion.json"
    long_doc = {
        "id": "long-doc",
        "title": "zh_enterprise_v1_seed:finance:示例公司:长文档",
        "path": "raw/long.pdf",
        "metadata": {"quality_status": "accepted_effect_long"},
    }
    clean_review = {
        "id": "clean-review",
        "title": "zh_enterprise_v1_seed:procurement:示例公司:采购制度",
        "path": "raw/review.pdf",
        "metadata": {
            "quality_status": "needs_case_review_short_but_usable",
            "source_org": "示例公司",
            "domain": "procurement",
            "doc_type": "procurement_policy",
            "source_format": "pdf",
            "cjk_chars": 4500,
            "page_count": 8,
        },
    }
    placeholder_review = {
        "id": "placeholder-review",
        "title": "zh_enterprise_v1_seed:procurement:待下载后从文档首页确认:采购公告",
        "path": "raw/placeholder.html",
        "metadata": {
            "quality_status": "needs_case_review_short_but_usable",
            "source_org": "待下载后从文档首页确认",
            "domain": "procurement",
            "doc_type": "notice",
            "source_format": "html",
            "cjk_chars": 6500,
            "page_count": None,
        },
    }
    long_manifest_path.write_text(json.dumps({"documents": [long_doc]}, ensure_ascii=False), encoding="utf-8")
    review_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "zh-enterprise-benchmark-manifest-v1",
                "documents": [long_doc, clean_review, placeholder_review],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ingestion_report_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "manifest_id": "clean-review",
                        "passed": True,
                        "chunk_count": 10,
                        "table_signal_chunk_count": 1,
                        "strict_noise_count": 0,
                        "noisy_chunk_count": 0,
                    },
                    {
                        "manifest_id": "placeholder-review",
                        "passed": True,
                        "chunk_count": 12,
                        "table_signal_chunk_count": 1,
                        "strict_noise_count": 0,
                        "noisy_chunk_count": 0,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report, manifest = module.build_review_promotion_report(
        long_manifest_path=long_manifest_path,
        review_manifest_path=review_manifest_path,
        ingestion_report_path=ingestion_report_path,
        promoted_dataset_name="zh_enterprise_promoted_test",
    )

    assert report["summary"]["promoted_review_document_count"] == 1
    assert report["summary"]["kept_review_document_count"] == 1
    assert report["summary"]["promoted_pilot_document_count"] == 2
    assert manifest["dataset_name"] == "zh_enterprise_promoted_test"
    assert {document["id"] for document in manifest["documents"]} == {"long-doc", "clean-review"}
    promoted = next(document for document in manifest["documents"] if document["id"] == "clean-review")
    assert promoted["metadata"]["quality_status"] == "promoted_review_pilot"
    kept = next(item for item in report["decisions"] if item["document_id"] == "placeholder-review")
    assert kept["decision"] == "keep_review"
    assert "source_org_placeholder" in kept["flags"]


def test_review_promotion_allows_html_and_below_4k_as_pilot_flags() -> None:
    module = _load_review_promotion_module()
    document = {
        "id": "html-review",
        "title": "zh_enterprise_v1_seed:procurement:示例公司:供应商资格预审公告",
        "metadata": {
            "source_org": "示例公司",
            "domain": "procurement",
            "doc_type": "supplier_notice",
            "source_format": "html",
            "cjk_chars": 3500,
        },
    }
    decision = module.classify_review_document(
        document,
        {
            "passed": True,
            "chunk_count": 8,
            "strict_noise_count": 0,
            "noisy_chunk_count": 0,
        },
    )

    assert decision["decision"] == "promote_to_pilot_effect"
    assert "html_source" in decision["flags"]
    assert "below_4k_cjk" in decision["flags"]
