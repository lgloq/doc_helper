from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


_LOCAL_BASELINE_ENV_NAMES = [
    "EMBEDDING_PROVIDER",
    "QUERY_REWRITE_PROVIDER",
    "RERANK_PROVIDER",
    "RETRIEVAL_HEURISTIC_RERANK_ENABLED",
    "RETRIEVAL_QUERY_PLAN_PROBE_ENABLED",
    "RETRIEVAL_VECTOR_ENABLED",
    "RERANK_MAX_CANDIDATES",
    "RETRIEVAL_DOCUMENT_DIVERSITY_MAX_CHUNKS",
    "RETRIEVAL_CJK_PYTHON_FALLBACK_MODE",
    "RETRIEVAL_CJK_PYTHON_SCORER",
    "RETRIEVAL_IN_DOCUMENT_EXPANSION_ENABLED",
    "RETRIEVAL_IN_DOCUMENT_EXPANSION_SEED_COUNT",
    "RETRIEVAL_IN_DOCUMENT_EXPANSION_PER_DOCUMENT",
    "RETRIEVAL_IN_DOCUMENT_EXPANSION_MAX_CANDIDATES",
    "RETRIEVAL_IN_DOCUMENT_EXPANSION_SCORE_WEIGHT",
    "RETRIEVAL_INDEXED_SPARSE_ENABLED",
    "RETRIEVAL_DOCUMENT_EVIDENCE_SWEEP_ENABLED",
    "RETRIEVAL_EVIDENCE_PRESERVATION_ENABLED",
    "RETRIEVAL_DOMAIN_PROFILE",
]


def _track_env(monkeypatch, env_names: list[str]) -> None:
    import os

    for env_name in env_names:
        if env_name in os.environ:
            monkeypatch.setenv(env_name, os.environ[env_name])
        else:
            monkeypatch.setenv(env_name, "")
            monkeypatch.delenv(env_name, raising=False)


def _load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stard_law_document_builder_keeps_clauses_inside_one_document() -> None:
    module = _load_script_module("import_benchmark_dataset.py")

    parsed = module.parse_stard_clause_name("中华人民共和国民法典第五十四条")
    title = module.stard_law_document_title("stard_zh_law_docs_small", parsed["law_name"])
    content = module.build_stard_law_document_text(
        parsed["law_name"],
        [
            {
                "id": "705",
                "article_name": parsed["article_name"],
                "clause_name": "中华人民共和国民法典第五十四条",
                "content": "自然人从事工商业经营，经依法登记，为个体工商户。",
            },
            {
                "id": "706",
                "article_name": "第五十五条",
                "clause_name": "中华人民共和国民法典第五十五条",
                "content": "农村集体经济组织的成员，依法取得农村土地承包经营权。",
            },
        ],
    )

    assert title == "stard_zh_law_docs_small:law:中华人民共和国民法典"
    assert content.count("## ") == 2
    assert "## 第五十四条" in content
    assert "条款全称：中华人民共和国民法典第五十四条" in content


def test_manifest_import_validates_file_sha256(tmp_path: Path) -> None:
    module = _load_script_module("import_benchmark_dataset.py")
    source_path = tmp_path / "policy.md"
    source_path.write_text("企业采购管理制度正文。", encoding="utf-8")
    spec = module.DocumentSpec(
        title="zh_enterprise_test:policy:采购管理制度",
        path=source_path,
        acl=[module.AclSpec(principal_type="public")],
        metadata={"file_sha256": "0" * 64},
    )

    with pytest.raises(ValueError, match="file_sha256 mismatch"):
        module.validate_specs([spec], [])

    spec.metadata["file_sha256"] = module.sha256_path(source_path)
    module.validate_specs([spec], [])


def test_manifest_import_document_slice_filters_cases() -> None:
    module = _load_script_module("import_benchmark_dataset.py")
    doc_a = module.DocumentSpec(title="dataset:doc:a", path=Path("a.md"))
    doc_b = module.DocumentSpec(title="dataset:doc:b", path=Path("b.md"))
    case_a = module.EvalCaseSpec(
        dataset_name="dataset",
        case_name="case:a",
        acting_user_email="viewer@local.test",
        question="A?",
        expected_document_titles=["dataset:doc:a"],
    )
    case_b = module.EvalCaseSpec(
        dataset_name="dataset",
        case_name="case:b",
        acting_user_email="viewer@local.test",
        question="B?",
        expected_document_titles=["dataset:doc:b"],
    )

    documents, cases = module.slice_manifest_import([doc_a, doc_b], [case_a, case_b], offset=1, limit=1)

    assert documents == [doc_b]
    assert cases == [case_b]


def test_manifest_import_reconcile_acl_parser_and_core_behavior(monkeypatch) -> None:
    module = _load_script_module("import_benchmark_dataset.py")
    args = module.build_parser().parse_args(["--reconcile-acl", "manifest", "--manifest", "manifest.json"])

    assert args.reconcile_acl is True

    keep_spec = module.AclSpec(principal_type="public")
    keep_entry = SimpleNamespace(
        principal_type=SimpleNamespace(value="public"),
        user_id=None,
        role_id=None,
        team_name=None,
        department_id=None,
    )
    stale_entry = SimpleNamespace(
        principal_type=SimpleNamespace(value="role"),
        user_id=None,
        role_id="manager-role-id",
        team_name=None,
        department_id=None,
    )
    deleted_entries = []
    fake_repository = SimpleNamespace(
        get_acl_entries=lambda document_id: [keep_entry, stale_entry],
        delete_acl_entry=deleted_entries.append,
    )
    fake_document_service = SimpleNamespace(document_repository=fake_repository)
    fake_session = SimpleNamespace(flushed=False, flush=lambda: setattr(fake_session, "flushed", True))
    monkeypatch.setattr(module, "acl_spec_key", lambda session, acl_spec: ("public", None, None, None, None))

    deleted = module.reconcile_acl_entries(fake_session, fake_document_service, "document-id", [keep_spec])

    assert deleted == 1
    assert deleted_entries == [stale_entry]
    assert fake_session.flushed is True


def test_retrieval_benchmark_scores_document_and_evidence_separately() -> None:
    module = _load_script_module("run_retrieval_benchmark.py")

    row = module.score_case(
        case=type(
            "Case",
            (),
            {
                "case_name": "stard:0",
                "acting_user_email": "viewer@local.test",
                "question": "谁可以成为个体工商户？",
            },
        )(),
        annotations={
            "expected_outcome": "answer",
            "expected_evidence_markers": [
                {
                    "label": "中华人民共和国民法典第五十四条",
                    "aliases": ["第五十四条", "自然人从事工商业经营"],
                    "document_title": "stard_zh_law_docs_small:law:中华人民共和国民法典",
                }
            ],
        },
        ranked_titles=["stard_zh_law_docs_small:law:中华人民共和国民法典"],
        ranked_chunks=[
            {
                "document_title": "stard_zh_law_docs_small:law:中华人民共和国民法典",
                "chunk_index": 3,
                "section_title": "第五十四条",
                "content": "条款全称：中华人民共和国民法典第五十四条\n\n自然人从事工商业经营，经依法登记，为个体工商户。",
            }
        ],
        expected_titles={"stard_zh_law_docs_small:law:中华人民共和国民法典"},
        forbidden_titles=set(),
        retrieval_debug={},
    )

    assert row["recall_at_k"] == 1.0
    assert row["evidence_recall_at_k"] == 1.0
    assert row["passed"] is True


def test_retrieval_benchmark_structural_marker_requires_structural_field_match() -> None:
    module = _load_script_module("run_retrieval_benchmark.py")

    row = module.score_case(
        case=type(
            "Case",
            (),
            {
                "case_name": "stard:structural-reference",
                "acting_user_email": "viewer@local.test",
                "question": "集体经营性建设用地如何安排？",
            },
        )(),
        annotations={
            "expected_outcome": "answer",
            "expected_evidence_markers": [
                {
                    "label": "土地管理法第六十三条",
                    "aliases": ["土地管理法第六十三条", "第六十三条"],
                    "document_title": "stard_zh_law_docs_small:law:土地管理法",
                }
            ],
        },
        ranked_titles=["stard_zh_law_docs_small:law:土地管理法"],
        ranked_chunks=[
            {
                "document_title": "stard_zh_law_docs_small:law:土地管理法",
                "chunk_index": 23,
                "section_title": "第二十三条",
                "clause_full_name": "土地管理法第二十三条",
                "article_number": "第二十三条",
                "content": "土地利用年度计划应当对本法第六十三条规定的集体经营性建设用地作出合理安排。",
            }
        ],
        expected_titles={"stard_zh_law_docs_small:law:土地管理法"},
        forbidden_titles=set(),
        retrieval_debug={},
    )

    assert row["recall_at_k"] == 1.0
    assert row["evidence_recall_at_k"] == 0.0
    assert row["passed"] is False
    assert row["missing_evidence_markers"] == ["土地管理法第六十三条"]


def test_retrieval_benchmark_fact_marker_still_matches_content() -> None:
    module = _load_script_module("run_retrieval_benchmark.py")

    rank = module.first_marker_rank(
        [
            {
                "document_title": "zh_enterprise_real:policy:样例管理办法",
                "chunk_index": 4,
                "section_title": "第四条",
                "clause_full_name": "样例管理办法第四条",
                "article_number": "第四条",
                "content": "样例管理遵循企业主体和政府引导的原则。",
            }
        ],
        {
            "label": "样例管理遵循企业主体和政府引导的原则",
            "aliases": ["样例管理遵循企业主体和政府引导的原则"],
            "document_title": "zh_enterprise_real:policy:样例管理办法",
        },
    )

    assert rank == 1


def test_retrieval_benchmark_long_marker_tolerates_pdf_title_noise() -> None:
    module = _load_script_module("run_retrieval_benchmark.py")

    rank = module.first_marker_rank(
        [
            {
                "document_title": "zh_enterprise_v1_seed:finance:测试募集说明书",
                "chunk_index": 104,
                "section_title": None,
                "clause_full_name": None,
                "article_number": None,
                "content": (
                    "测试募集说明书 1、关于涉及战略重组的提示性公告发行人于2021年7月15日发布"
                    "《测试公司关于涉及战略重组的提示性公告》指出，发行人于2021年7月14日收到"
                    "控股股东通知，相关机构正筹划对发行人战略重组事项。"
                ),
            }
        ],
        {
            "label": (
                "（十三）战略重组情况 44 测试公司2025年度中期票据募集说明书 1、关于涉及战略重组的提示性公告"
                "发行人于2021年7月15日发布《测试公司关于涉及战略重组的提示性公告》指出，"
                "发行人于2021年7月14日收到控股股东通知，相关机构正筹划对发行人战略重组事项。"
            ),
            "aliases": [],
            "document_title": "zh_enterprise_v1_seed:finance:测试募集说明书",
        },
    )

    assert rank == 1


def test_retrieval_benchmark_long_marker_fuzzy_match_rejects_sparse_overlap() -> None:
    module = _load_script_module("run_retrieval_benchmark.py")

    rank = module.first_marker_rank(
        [
            {
                "document_title": "zh_enterprise_v1_seed:ipo:样例上市公告书",
                "chunk_index": 10,
                "section_title": None,
                "clause_full_name": None,
                "article_number": None,
                "content": (
                    "截至2025年9月30日，公司核心产品已在国内市场销售多年，市场接受度较高，"
                    "国产厂家陆续获批上市。"
                ),
            }
        ],
        {
            "label": (
                "（1）截至2025年9月30日，公司无持股比例达到30%的单一股东，发行人股权较为分散，"
                "宋亮直接持有发行人16.1530%的股份，并通过多个员工持股平台合计控制公司13.4140%的股份。"
            ),
            "aliases": [],
            "document_title": "zh_enterprise_v1_seed:ipo:样例上市公告书",
        },
    )

    assert rank is None


def test_retrieval_benchmark_local_baseline_sets_domain_profile(monkeypatch) -> None:
    module = _load_script_module("run_retrieval_benchmark.py")
    _track_env(monkeypatch, _LOCAL_BASELINE_ENV_NAMES)

    module.apply_local_baseline_env("legal_benchmark")

    assert module.os.environ["RETRIEVAL_DOMAIN_PROFILE"] == "legal_benchmark"
    assert module.os.environ["RETRIEVAL_HEURISTIC_RERANK_ENABLED"] == "false"


def test_retrieval_benchmark_applies_cjk_python_ablation_overrides(monkeypatch) -> None:
    module = _load_script_module("run_retrieval_benchmark.py")
    _track_env(monkeypatch, _LOCAL_BASELINE_ENV_NAMES)

    module.apply_local_baseline_env("enterprise")
    module.apply_retrieval_ablation_overrides(cjk_python_fallback_mode="always", cjk_python_scorer="bm25")

    assert module.os.environ["RETRIEVAL_CJK_PYTHON_FALLBACK_MODE"] == "always"
    assert module.os.environ["RETRIEVAL_CJK_PYTHON_SCORER"] == "bm25"


def test_benchmark_eval_loads_manifest_case_names(tmp_path: Path) -> None:
    module = _load_script_module("run_benchmark_eval.py")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_name": "case:a"},
                    {"case_name": "case:b"},
                    {"case_name": "case:a"},
                    {"case_name": ""},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert module.load_manifest_case_names(str(manifest_path)) == ["case:a", "case:b"]


def test_benchmark_eval_applies_full_indexed_sparse_env(monkeypatch) -> None:
    module = _load_script_module("run_benchmark_eval.py")
    _track_env(monkeypatch, _LOCAL_BASELINE_ENV_NAMES + ["RETRIEVAL_LEXICAL_ENABLED", "RETRIEVAL_STRUCTURAL_ENABLED"])

    module.apply_retrieval_ablation_env("full_indexed_sparse")

    assert module.os.environ["RETRIEVAL_INDEXED_SPARSE_ENABLED"] == "true"
    assert module.os.environ["RETRIEVAL_LEXICAL_ENABLED"] == "true"
    assert module.os.environ["RETRIEVAL_STRUCTURAL_ENABLED"] == "true"
    assert module.os.environ["RETRIEVAL_VECTOR_ENABLED"] == "false"


def test_retrieval_benchmark_summary_includes_latency_percentiles() -> None:
    module = _load_script_module("run_retrieval_benchmark.py")
    rows = [
        {
            "passed": True,
            "recall_at_k": 1.0,
            "precision_at_k": 1.0,
            "average_precision_at_k": 1.0,
            "mrr": 1.0,
            "ndcg_at_k": 1.0,
            "evidence_recall_at_k": 1.0,
            "evidence_mrr": 1.0,
            "permission_isolation_correct": True,
            "expected_outcome": "answer",
            "failure_mode": "passed",
            "elapsed_seconds": 1.0,
            "case_name": "case:fast",
            "retrieval_debug": {"search_total_latency_ms": 900},
        },
        {
            "passed": False,
            "recall_at_k": 1.0,
            "precision_at_k": 0.5,
            "average_precision_at_k": 1.0,
            "mrr": 1.0,
            "ndcg_at_k": 1.0,
            "evidence_recall_at_k": 0.0,
            "evidence_mrr": 0.0,
            "permission_isolation_correct": True,
            "expected_outcome": "answer",
            "failure_mode": "expected_evidence_missing",
            "elapsed_seconds": 5.0,
            "case_name": "case:slow",
            "retrieval_debug": {"search_total_latency_ms": 4900},
        },
    ]

    summary = module.summarize(rows)
    slowest = module.slowest_cases(rows, limit=1)

    assert summary["latency_seconds"]["case_count"] == 2
    assert summary["latency_seconds"]["p50"] == 3.0
    assert summary["latency_seconds"]["max"] == 5.0
    assert slowest == [
        {
            "case_name": "case:slow",
            "elapsed_seconds": 5.0,
            "passed": False,
            "failure_mode": "expected_evidence_missing",
            "search_total_latency_ms": 4900,
        }
    ]


def test_retrieval_ablation_report_renders_latency_and_slowest_cases() -> None:
    module = _load_script_module("run_retrieval_ablation_benchmark.py")
    report = {
        "dataset_name": "zh_enterprise_test",
        "manifest": "manifest.json",
        "sample": {
            "selected_case_count": 1,
            "selected_case_type_counts": {"single_fact": 1},
        },
        "top_k": 10,
        "ablations": [
            {
                "ablation": {"name": "indexed_sparse_only"},
                "summary": {
                    "pass_rate": 0.95,
                    "recall_at_k_avg": 1.0,
                    "evidence_recall_at_k_avg": 0.97,
                    "permission_isolation_pass_rate": 1.0,
                },
                "latency": {"p95": 1.23, "max": 2.34},
                "slowest_cases": [
                    {
                        "case_name": "case:slow",
                        "elapsed_seconds": 2.34,
                        "passed": False,
                        "failure_mode": "expected_evidence_missing",
                        "search_total_latency_ms": 2300,
                    }
                ],
                "failure_mode_counts": {"expected_evidence_missing": 1},
            }
        ],
    }

    markdown = module.render_markdown(report)
    summary = module.summary_text(report)

    assert "P95 latency" in markdown
    assert "Slowest Cases: indexed_sparse_only" in markdown
    assert "case:slow" in markdown
    assert "p95=1.23 max=2.34" in summary


def test_latency_outlier_audit_uses_per_case_retrieval_debug() -> None:
    module = _load_script_module("audit_retrieval_latency_outliers.py")
    report = {
        "dataset_name": "zh_enterprise_test",
        "retrieval_domain_profile": "enterprise",
        "cases": [
            {
                "case_name": "case:slow-indexed",
                "question": "请核对测试公司披露事项。",
                "passed": True,
                "failure_mode": "passed",
                "elapsed_seconds": 9.5,
                "retrieval_debug": {
                    "search_total_latency_ms": 9500,
                    "lexical_retrieval_latency_ms": 100,
                    "indexed_sparse_retrieval_latency_ms": 9100,
                    "rerank_latency_ms": 20,
                    "indexed_sparse_candidate_count": 80,
                    "query_decomposition_applied": True,
                    "subquery_count": 2,
                    "subquery_timeout_count": 0,
                },
            }
        ],
    }

    audit = module.audit_latency_outliers(report, slow_threshold_ms=8000, top_n=5)
    outlier = audit["ablations"][0]["outliers"][0]

    assert audit["summary"]["debug_available_count"] == 1
    assert audit["summary"]["debug_missing_count"] == 0
    assert outlier["dominant_stage"] == "indexed_sparse_retrieval"
    assert outlier["stage_latencies_ms"][0] == {"stage": "indexed_sparse_retrieval", "latency_ms": 9100}
    assert outlier["candidate_counts"][0] == {"stage": "indexed_sparse", "count": 80}
    assert outlier["action_hint"] == "inspect_sparse_query_breadth_cjk_fallback_and_statement_timeout"


def test_latency_outlier_audit_recommends_replay_when_ablation_summary_lacks_debug() -> None:
    module = _load_script_module("audit_retrieval_latency_outliers.py")
    report = {
        "dataset_name": "zh_enterprise_test",
        "ablations": [
            {
                "ablation": {"name": "full_indexed_sparse"},
                "summary": {"pass_rate": 1.0},
                "slowest_cases": [
                    {
                        "case_name": "case:slow-permission",
                        "elapsed_seconds": 62.9,
                        "passed": True,
                        "failure_mode": "passed",
                        "search_total_latency_ms": 62900,
                    }
                ],
                "failure_cases": [],
            }
        ],
    }

    audit = module.audit_latency_outliers(report, slow_threshold_ms=8000, top_n=5)
    outlier = audit["ablations"][0]["outliers"][0]
    markdown = module.render_markdown(audit)

    assert audit["summary"]["debug_missing_count"] == 1
    assert audit["recommended_replay_cases"][0]["case_name"] == "case:slow-permission"
    assert outlier["dominant_stage"] == "debug_missing"
    assert outlier["action_hint"] == "replay_target_case_with_full_retrieval_debug"
    assert "Recommended Targeted Replays" in markdown


def test_retrieval_benchmark_manifest_scope_and_exact_case_filter(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module("run_retrieval_benchmark.py")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"documents":[{"title":"zh_enterprise_test:policy:Doc B"}]}',
        encoding="utf-8",
    )
    cases = [
        SimpleNamespace(
            case_name="case:a",
            acting_user_email="viewer@local.test",
            question="A?",
            forbidden_document_titles=[],
        ),
        SimpleNamespace(
            case_name="case:b",
            acting_user_email="viewer@local.test",
            question="B?",
            forbidden_document_titles=[],
        ),
    ]
    search_calls = []

    class FakeSession:
        def execute(self, _statement):
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(id="doc-b-id", title="zh_enterprise_test:policy:Doc B"),
                ]
            )

        def close(self):
            pass

    class FakeEvalRepository:
        def __init__(self, _session):
            pass

        def list_cases(self, dataset_name):
            assert dataset_name == "zh_enterprise_test"
            return cases

    class FakeUserRepository:
        def __init__(self, _session):
            pass

        def get_by_email(self, email):
            assert email == "viewer@local.test"
            return SimpleNamespace(email=email)

    class FakeRetrievalService:
        def __init__(self, _session):
            self.settings = SimpleNamespace(effective_retrieval_domain_profile="enterprise")

        def search(self, _actor, request, *, scoped_document_ids):
            search_calls.append({"query": request.query, "scoped_document_ids": scoped_document_ids})
            return SimpleNamespace(
                matched_chunks=[
                    SimpleNamespace(
                        document_title="zh_enterprise_test:policy:Doc B",
                        chunk_index=0,
                        section_title=None,
                        clause_full_name=None,
                        article_number=None,
                        chunk_type="paragraph",
                        content="Doc B evidence",
                    )
                ],
                debug=SimpleNamespace(model_dump=lambda: {"scoped": scoped_document_ids}),
            )

    monkeypatch.setattr(module, "seed_mock_data", lambda: None)
    monkeypatch.setattr(module, "SessionLocal", FakeSession)
    monkeypatch.setattr(module, "EvalRepository", FakeEvalRepository)
    monkeypatch.setattr(module, "UserRepository", FakeUserRepository)
    monkeypatch.setattr(module, "RetrievalService", FakeRetrievalService)
    monkeypatch.setattr(
        module.EvalService,
        "_resolve_case_annotations",
        staticmethod(
            lambda case: {
                "expected_outcome": "answer",
                "expected_retrieval_titles": ["zh_enterprise_test:policy:Doc B"],
                "expected_evidence_markers": [],
            }
        ),
    )

    report = module.run_retrieval_benchmark(
        dataset_name="zh_enterprise_test",
        top_k=5,
        limit=None,
        case_names=["case:b"],
        case_name_contains=None,
        document_title_prefix=None,
        manifest_scope=str(manifest_path),
    )

    assert [row["case_name"] for row in report["cases"]] == ["case:b"]
    assert search_calls == [{"query": "B?", "scoped_document_ids": ["doc-b-id"]}]
    assert report["document_scope"]["mode"] == "manifest"
    assert report["document_scope"]["manifest_document_count"] == 1


def test_backfill_chunk_lexical_search_text_parser_defaults() -> None:
    module = _load_script_module("backfill_chunk_lexical_search_text.py")

    parser = module.build_parser()
    args = parser.parse_args([])

    assert args.batch_size == 500
    assert args.document_title_prefix is None
    assert args.rebuild_all is False


def test_backfill_chunk_lexical_search_text_blank_detection() -> None:
    module = _load_script_module("backfill_chunk_lexical_search_text.py")

    assert module._has_search_text("客户响应") is True
    assert module._has_search_text("") is False
    assert module._has_search_text("   ") is False
    assert module._has_search_text(None) is False

def test_backfill_missing_chunk_embeddings_parser_defaults() -> None:
    module = _load_script_module("backfill_missing_chunk_embeddings.py")

    parser = module.build_parser()
    args = parser.parse_args([])

    assert args.title_prefix == "zh_enterprise_v1_seed:%"
    assert args.batch_size == 128
    assert args.limit is None
    assert args.dry_run is False
    assert "missing_before=2" in module.summary_text(
        {
            "passed": False,
            "dry_run": True,
            "missing_before": 2,
            "updated_chunks": 0,
            "missing_after": 2,
        }
    )


def test_zh_enterprise_manifest_uses_clean_html_and_evidence_markers(tmp_path, monkeypatch) -> None:
    module = _load_script_module("build_zh_enterprise_benchmark.py")
    source = module.SourceDocument(
        id="sample_policy",
        title="zh_enterprise:policy:样例管理办法",
        url="https://example.com/sample.htm",
        description="样例 HTML。",
        expected_facts=["样例管理遵循企业主体和政府引导的原则"],
        questions=["样例管理遵循什么原则？"],
        acl=[{"principal_type": "public"}],
        metadata={"domain": "policy", "source_org": "样例机构", "language": "zh"},
    )
    monkeypatch.setattr(module, "SOURCES", [source])
    monkeypatch.setattr(module, "DISTRACTOR_SOURCES", [])
    raw_path = module.raw_html_source_path(tmp_path, source)
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        "<html><body><nav>登录 注册 首页</nav><h1>样例管理办法</h1>"
        "<p>样例管理遵循企业主体和政府引导的原则。</p>"
        "<footer>责任编辑：测试</footer></body></html>",
        encoding="utf-8",
    )

    manifest = module.build_manifest(tmp_path, download=False, dataset_name="zh_enterprise_test", user_agent="test")
    clean_path = tmp_path / manifest["documents"][0]["path"]
    case = manifest["cases"][0]

    assert clean_path.suffix == ".md"
    assert "登录" not in clean_path.read_text(encoding="utf-8")
    assert case["expected_evidence_markers"]
    assert case["expected_evidence_markers"][0]["document_title"] == "zh_enterprise_test:policy:样例管理办法"
    assert len(case["expected_evidence_markers"]) == 1


def test_zh_enterprise_case_manifest_emits_locators_and_excludes_format_coverage() -> None:
    module = _load_script_module("build_zh_enterprise_case_manifest.py")
    effect_info = module.DocumentInfo(
        id="effect_doc",
        title="zh_enterprise_test:policy:有效制度",
        domain="internal_control",
        doc_type="policy",
        source_org="测试公司",
        source_format="md",
        benchmark_role="effect",
        restricted=False,
    )
    format_only_info = module.DocumentInfo(
        id="format_doc",
        title="zh_enterprise_test:format:短网页",
        domain="procurement",
        doc_type="notice",
        source_org="测试公司",
        source_format="html",
        benchmark_role="format_coverage_only",
        restricted=False,
    )
    effect_candidate = module.EvidenceCandidate(
        key="effect:1",
        doc_id="effect_doc",
        document_title=effect_info.title,
        domain=effect_info.domain,
        source_org=effect_info.source_org,
        chunk_index=7,
        chunk_type="paragraph",
        section_title="第三条",
        heading_path="第一章/第三条",
        page=2,
        paragraph_index=4,
        quote="内部控制整改应当明确责任部门、完成时限，并持续跟踪整改效果。",
        topic="内控合规和风险管理",
        score=100,
    )
    format_only_candidate = module.EvidenceCandidate(
        key="format:1",
        doc_id="format_doc",
        document_title=format_only_info.title,
        domain=format_only_info.domain,
        source_org=format_only_info.source_org,
        chunk_index=1,
        chunk_type="paragraph",
        section_title=None,
        heading_path=None,
        page=None,
        paragraph_index=None,
        quote="短网页只用于格式覆盖，不应进入效果类检索案例生成池。",
        topic="关键披露或管理要求",
        score=90,
    )
    builder = module.CaseBuilder(
        {"effect_doc": effect_info},
        [effect_candidate],
        restricted_doc_ids=set(),
        per_doc_limit=3,
    )

    case = builder.build_single_fact_cases(1)[0]
    marker = case["expected_evidence_markers"][0]

    assert marker["document_id"] == "effect_doc"
    assert marker["source_chunk_index"] == 7
    assert marker["evidence_locator"]["text_quote"] == effect_candidate.quote
    assert marker["evidence_locator"]["section_title"] == "第三条"
    assert marker["evidence_locator"]["page"] == 2
    assert module.is_format_coverage_only_document(
        {"path": "short.html", "metadata": {"benchmark_role": "format_coverage_only", "cjk_chars": 2000}}
    )
    assert format_only_candidate.doc_id not in case["expected_document_ids"]


def test_zh_enterprise_low_overlap_question_keeps_concrete_evidence_anchor() -> None:
    module = _load_script_module("build_zh_enterprise_case_manifest.py")
    candidate = module.EvidenceCandidate(
        key="effect:2",
        doc_id="effect_doc",
        document_title="zh_enterprise_test:finance:融资文件",
        domain="finance",
        source_org="测试公司",
        chunk_index=8,
        chunk_type="paragraph",
        section_title=None,
        heading_path=None,
        page=None,
        paragraph_index=None,
        quote="（七）关于基础募集说明书第五章发行人治理结构与内控制度的更新，公司严格按照公司法完善内部控制制度体系。",
        topic="融资安排与偿债披露",
        score=120,
    )

    question = module.low_overlap_question(candidate)

    assert "融资安排与偿债披露" in question
    assert "重点核对" in question
    assert "关于基础募集说明书第五章发行人治理结构与内控制度的更新" in question


def test_benchmark_case_quality_audit_flags_broad_low_overlap_cases() -> None:
    module = _load_script_module("audit_benchmark_case_quality.py")
    manifest = {
        "dataset_name": "quality_test",
        "cases": [
            {
                "case_name": "case:broad",
                "question": "投研团队准备底稿时，需要在测试公司的融资与财务披露材料里确认“融资安排与偿债披露”相关事项的处理口径。请指出相关原文依据。",
                "expected_outcome": "answer",
                "expected_evidence_markers": [
                    {
                        "label": "测试公司融资安排与偿债披露基础信息后，另行说明突发事件应急管理办法和预防措施。",
                        "aliases": ["突发事件应急管理办法"],
                    }
                ],
                "metadata": {"case_type": "low_overlap_enterprise_scenario"},
            },
            {
                "case_name": "case:anchored_scenario",
                "question": (
                    "投研团队准备底稿时，需要在测试公司的融资与财务披露材料里确认"
                    "“融资安排与偿债披露，重点核对“内部控制制度体系””相关事项的处理口径。"
                    "请指出相关原文依据。"
                ),
                "expected_outcome": "answer",
                "expected_evidence_markers": [
                    {
                        "label": "公司严格按照公司法完善内部控制制度体系，董事会及相关决策机构规范运作。",
                        "aliases": ["内部控制制度体系"],
                    }
                ],
                "metadata": {"case_type": "low_overlap_enterprise_scenario"},
            },
            {
                "case_name": "case:anchored",
                "question": "测试公司的融资与财务披露材料中，“内部控制制度体系”具体是怎么披露的？",
                "expected_outcome": "answer",
                "expected_evidence_markers": [
                    {
                        "label": "公司严格按照公司法完善内部控制制度体系，董事会及相关决策机构规范运作。",
                        "aliases": ["内部控制制度体系"],
                    }
                ],
                "metadata": {"case_type": "single_fact"},
            },
            {
                "case_name": "case:weak_anchor",
                "question": (
                    "比较甲公司和乙公司两份招股与上市申报材料，分别关注"
                    "“担保情况”和“截至2025年9月30日”，各引用一处原文依据。"
                ),
                "expected_outcome": "answer",
                "expected_evidence_markers": [
                    {"label": "担保情况为实际控制人提供连带责任保证。", "aliases": ["担保情况"]},
                    {
                        "label": "截至2025年9月30日，公司无持股比例达到30%的单一股东。",
                        "aliases": ["截至2025年9月30日"],
                    },
                ],
                "metadata": {"case_type": "multi_evidence_cross_document"},
            },
        ],
    }

    report = module.audit_manifest_case_quality(manifest)
    by_name = {item["case_name"]: item for item in report["cases"]}
    annotated = module.build_annotated_manifest(report, manifest)
    strict = module.build_strict_manifest(report, manifest)

    assert by_name["case:broad"]["metric_group"] == "broad_document_discovery"
    assert "broad_question_with_exact_gold" in by_name["case:broad"]["flags"]
    assert "missing_concrete_evidence_anchor" in by_name["case:broad"]["flags"]
    assert by_name["case:anchored_scenario"]["metric_group"] == "strict_exact_evidence"
    assert by_name["case:anchored"]["metric_group"] == "strict_exact_evidence"
    assert by_name["case:weak_anchor"]["metric_group"] == "anchor_quality_review"
    assert "unreliable_question_anchor" in by_name["case:weak_anchor"]["flags"]
    assert "weak_question_anchor:too_short" in by_name["case:weak_anchor"]["flags"]
    assert "weak_question_anchor:date_only" in by_name["case:weak_anchor"]["flags"]
    assert "weak_question_evidence_overlap" not in by_name["case:anchored"]["flags"]
    assert annotated["cases"][0]["metadata"]["quality_audit"]["metric_group"] == "broad_document_discovery"
    assert [case["case_name"] for case in strict["cases"]] == ["case:anchored_scenario", "case:anchored"]
    assert strict["case_quality_audit"]["excluded_case_count"] == 2


def test_benchmark_case_quality_audit_excludes_marker_missing_in_chunks() -> None:
    module = _load_script_module("audit_benchmark_case_quality.py")
    manifest = {
        "dataset_name": "quality_test",
        "cases": [
            {
                "case_name": "case:valid",
                "question": "测试公司的制度中，“审批流程应由负责人复核”具体是怎么规定的？",
                "expected_outcome": "answer",
                "expected_evidence_markers": [{"label": "审批流程应由负责人复核。", "aliases": ["审批流程应由负责人复核"]}],
                "metadata": {"case_type": "single_fact"},
            },
            {
                "case_name": "case:missing_marker",
                "question": "测试公司的制度中，“募集资金投资计划使用”具体是怎么规定的？",
                "expected_outcome": "answer",
                "expected_evidence_markers": [{"label": "保证募集资金按照发行申请文件中承诺的募集资金投资计划使用"}],
                "metadata": {"case_type": "single_fact"},
            },
        ],
    }
    marker_presence_issues = {
        "case:missing_marker": {
            "root_cause": "marker_not_found_in_expected_document_chunks",
            "missing_marker_count": 1,
            "missing_marker_preview": ["保证募集资金按照发行申请文件中承诺的募集资金投资计划使用"],
            "source_reports": ["root-cause.json"],
        }
    }

    report = module.audit_manifest_case_quality(manifest, marker_presence_issues=marker_presence_issues)
    by_name = {item["case_name"]: item for item in report["cases"]}
    strict = module.build_strict_manifest(report, manifest)

    assert by_name["case:missing_marker"]["metric_group"] == "anchor_quality_review"
    assert "expected_marker_not_found_in_chunks" in by_name["case:missing_marker"]["flags"]
    assert by_name["case:missing_marker"]["marker_presence_issue"]["missing_marker_count"] == 1
    assert [case["case_name"] for case in strict["cases"]] == ["case:valid"]


def test_benchmark_case_quality_audit_flags_period_anchor_missing_gold_discriminator() -> None:
    module = _load_script_module("audit_benchmark_case_quality.py")
    manifest = {
        "dataset_name": "quality_test",
        "cases": [
            {
                "case_name": "case:period_anchor",
                "question": (
                    "请同时核对北京城建投资发展股份有限公司这份融资与财务披露材料中的两个事项："
                    "“可能存在减值的风险”和“截至 2025 年末”，分别引用依据。"
                ),
                "expected_outcome": "answer",
                "expected_document_ids": ["doc:beijing"],
                "expected_evidence_markers": [
                    {
                        "label": (
                            "截至 2025 年末，发行人在建项目收入确认情况正常，对于房屋开发项目，"
                            "在开发产品已经竣工且经有关部门验收合格达到可交付状态。"
                        ),
                        "aliases": [
                            "截至 2025 年末，发行人在建项目收入确认情况正常，对于房屋开发项目"
                        ],
                        "document_id": "doc:beijing",
                        "document_title": "北京城建募集说明书",
                    }
                ],
                "metadata": {"case_type": "multi_evidence_same_document"},
            }
        ],
    }

    report = module.audit_manifest_case_quality(manifest)
    case_report = report["cases"][0]

    assert case_report["metric_group"] == "anchor_quality_review"
    assert "unreliable_question_anchor" in case_report["flags"]
    assert "under_specified_question_anchor" in case_report["flags"]
    assert "weak_question_anchor:period_only" in case_report["flags"]
    assert "weak_question_anchor:missing_gold_discriminator" in case_report["flags"]
    assert case_report["missing_gold_discriminators"][0]["missing_phrase"] == "在建项目收入确认情况正常"


def test_benchmark_case_quality_audit_counts_repeated_anchor_in_expected_document_text() -> None:
    module = _load_script_module("audit_benchmark_case_quality.py")
    manifest = {
        "dataset_name": "quality_test",
        "cases": [
            {
                "case_name": "case:repeated_anchor",
                "question": "测试公司的年报中，“截至 2025 年末”具体是怎么披露的？",
                "expected_outcome": "answer",
                "expected_document_ids": ["doc:annual"],
                "expected_evidence_markers": [
                    {
                        "label": "截至 2025 年末，公司核心系统改造项目已完成验收。",
                        "document_id": "doc:annual",
                    }
                ],
                "metadata": {"case_type": "single_fact"},
            }
        ],
    }
    document_texts = {
        "doc:annual": (
            "截至 2025 年末，资产总额增加。截至 2025 年末，负债率下降。"
            "截至 2025 年末，现金余额改善。截至 2025 年末，核心系统改造项目已完成验收。"
        )
    }

    report = module.audit_manifest_case_quality(
        manifest,
        document_texts=document_texts,
        anchor_occurrence_threshold=3,
    )
    case_report = report["cases"][0]

    assert "weak_question_anchor:repeated_in_expected_document" in case_report["flags"]
    assert case_report["anchor_occurrence_stats"][0]["max_document_count"] == 4
    assert case_report["anchor_occurrence_stats"][0]["documents_checked"] == 1
