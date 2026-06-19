from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4


_ABLATION_ENV_NAMES = [
    "RETRIEVAL_LEXICAL_ENABLED",
    "RETRIEVAL_VECTOR_ENABLED",
    "RETRIEVAL_INDEXED_SPARSE_ENABLED",
    "RETRIEVAL_STRUCTURAL_ENABLED",
    "RETRIEVAL_IN_DOCUMENT_EXPANSION_ENABLED",
    "RETRIEVAL_DOCUMENT_EVIDENCE_SWEEP_ENABLED",
    "RETRIEVAL_DOCUMENT_FIRST_EVIDENCE_ENABLED",
    "RETRIEVAL_DOCUMENT_NEIGHBOR_CONTEXT_ENABLED",
    "RETRIEVAL_EVIDENCE_PRESERVATION_ENABLED",
    "RETRIEVAL_FINAL_COVERAGE_ENABLED",
    "RETRIEVAL_EVIDENCE_QUERY_BRIDGE_ENABLED",
    "RETRIEVAL_HEURISTIC_RERANK_ENABLED",
    "RERANK_PROVIDER",
]


def _track_env(monkeypatch, env_names: list[str]) -> None:
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


def test_retrieval_diagnostics_parses_named_ablations() -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")

    ablations = module.parse_ablations(
        "full_local,full_heuristic_rerank,indexed_sparse_only,vector_only,full_document_first_evidence_bridge,full_document_first_neighbor"
    )

    assert [item.name for item in ablations] == [
        "full_local",
        "full_heuristic_rerank",
        "indexed_sparse_only",
        "vector_only",
        "full_document_first_evidence_bridge",
        "full_document_first_neighbor",
    ]
    assert ablations[0].use_structural is True
    assert ablations[1].force_heuristic_rerank is True
    assert ablations[2].use_structural is False
    assert ablations[2].use_indexed_sparse is True
    assert ablations[2].use_lexical is False
    assert ablations[3].use_vector is True
    assert ablations[4].use_document_first is True
    assert ablations[4].use_evidence_bridge is True
    assert ablations[5].use_document_first is True
    assert ablations[5].use_neighbor is True


def test_retrieval_diagnostics_applies_lexical_ablation_env(monkeypatch) -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")
    _track_env(monkeypatch, _ABLATION_ENV_NAMES)

    module.apply_ablation_env(module.parse_ablations("indexed_sparse_only")[0])
    assert os.environ["RETRIEVAL_LEXICAL_ENABLED"] == "false"

    module.apply_ablation_env(module.parse_ablations("full_indexed_sparse")[0])
    assert os.environ["RETRIEVAL_LEXICAL_ENABLED"] == "true"


def test_retrieval_diagnostics_filters_exact_case_names() -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")
    cases = [
        SimpleNamespace(case_name="stard:1"),
        SimpleNamespace(case_name="stard:10"),
        SimpleNamespace(case_name="stard:11"),
    ]

    selected = module.filter_cases(
        cases,
        case_names=["stard:10", "stard:11"],
        case_name_contains=None,
        limit=None,
    )

    assert [case.case_name for case in selected] == ["stard:10", "stard:11"]


def test_retrieval_diagnostics_parses_rerank_result_limit() -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")

    args = module.build_parser().parse_args(
        [
            "--dataset",
            "demo",
            "--rerank-result-limit",
            "120",
        ]
    )

    assert args.rerank_result_limit == 120


def test_root_cause_audit_parses_rerank_result_limit() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")

    args = module.build_parser().parse_args(
        [
            "--dataset",
            "demo",
            "--output",
            "out.json",
            "--rerank-result-limit",
            "200",
        ]
    )

    assert args.rerank_result_limit == 200


def test_root_cause_audit_parses_force_heuristic_rerank() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")

    args = module.build_parser().parse_args(
        [
            "--dataset",
            "demo",
            "--output",
            "out.json",
            "--force-heuristic-rerank",
        ]
    )

    assert args.force_heuristic_rerank is True


def test_retrieval_diagnostics_applies_in_document_expansion_overrides(monkeypatch) -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")
    env_names = [
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_SEED_COUNT",
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_PER_DOCUMENT",
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_MAX_CANDIDATES",
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_ADJACENT_WINDOW",
        "RETRIEVAL_IN_DOCUMENT_EXPANSION_SCORE_WEIGHT",
        "RERANK_MAX_CANDIDATES",
        "RETRIEVAL_HEURISTIC_RERANK_ENABLED",
    ]
    for env_name in env_names:
        monkeypatch.delenv(env_name, raising=False)

    args = module.build_parser().parse_args(
        [
            "--dataset",
            "demo",
            "--in-document-expansion-seed-count",
            "80",
            "--in-document-expansion-per-document",
            "10",
            "--in-document-expansion-max-candidates",
            "96",
            "--in-document-expansion-adjacent-window",
            "20",
            "--in-document-expansion-score-weight",
            "0.8",
            "--rerank-max-candidates",
            "64",
        ]
    )

    module.apply_in_document_expansion_overrides(args)

    assert os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_SEED_COUNT"] == "80"
    assert os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_PER_DOCUMENT"] == "10"
    assert os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_MAX_CANDIDATES"] == "96"
    assert os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_ADJACENT_WINDOW"] == "20"
    assert os.environ["RETRIEVAL_IN_DOCUMENT_EXPANSION_SCORE_WEIGHT"] == "0.8"
    assert os.environ["RERANK_MAX_CANDIDATES"] == "64"
    for env_name in env_names:
        monkeypatch.delenv(env_name, raising=False)


def test_retrieval_diagnostics_classifies_source_pool_evidence_missing() -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")

    stage_scores = {
        "lexical": {"title_recall": 1.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "structural": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "vector": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "fused": {"evidence_recall": 0.0},
        "pre_rerank": {"evidence_recall": 0.0},
        "reranked": {"evidence_recall": 0.0},
        "final": {"evidence_recall": 0.0},
    }
    final_score = {"passed": False, "expected_outcome": "answer", "failure_mode": "expected_evidence_missing"}

    assert module.classify_stage_loss(stage_scores, final_score) == "source_pool_evidence_missing"


def test_retrieval_diagnostics_classifies_final_selection_drop() -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")

    stage_scores = {
        "lexical": {"title_recall": 1.0, "evidence_recall": 1.0, "matched_evidence_markers": ["marker"]},
        "structural": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "vector": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "fused": {"evidence_recall": 1.0},
        "pre_rerank": {"evidence_recall": 1.0},
        "reranked": {"evidence_recall": 1.0},
        "final": {"evidence_recall": 0.0},
    }
    final_score = {"passed": False, "expected_outcome": "answer", "failure_mode": "expected_evidence_missing"}

    assert module.classify_stage_loss(stage_scores, final_score) == "final_selection_evidence_dropped"


def test_retrieval_diagnostics_counts_document_sweep_as_source_pool() -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")

    stage_scores = {
        "lexical": {"title_recall": 1.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "structural": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "vector": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "document_sweep": {"title_recall": 1.0, "evidence_recall": 1.0, "matched_evidence_markers": ["marker"]},
        "fused": {"evidence_recall": 1.0},
        "pre_rerank": {"evidence_recall": 1.0},
        "reranked": {"evidence_recall": 1.0},
        "final": {"evidence_recall": 0.0},
    }
    final_score = {"passed": False, "expected_outcome": "answer", "failure_mode": "expected_evidence_missing"}

    assert module.classify_stage_loss(stage_scores, final_score) == "final_selection_evidence_dropped"


def test_retrieval_diagnostics_counts_indexed_sparse_as_source_pool() -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")

    stage_scores = {
        "lexical": {"title_recall": 1.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "indexed_sparse": {"title_recall": 1.0, "evidence_recall": 1.0, "matched_evidence_markers": ["marker"]},
        "structural": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "vector": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "fused": {"evidence_recall": 1.0},
        "pre_rerank": {"evidence_recall": 1.0},
        "reranked": {"evidence_recall": 1.0},
        "final": {"evidence_recall": 0.0},
    }
    final_score = {"passed": False, "expected_outcome": "answer", "failure_mode": "expected_evidence_missing"}

    assert module.classify_stage_loss(stage_scores, final_score) == "final_selection_evidence_dropped"


def test_retrieval_diagnostics_classifies_preservation_pool_drop() -> None:
    module = _load_script_module("run_retrieval_diagnostics.py")

    stage_scores = {
        "lexical": {"title_recall": 1.0, "evidence_recall": 1.0, "matched_evidence_markers": ["marker"]},
        "structural": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "vector": {"title_recall": 0.0, "evidence_recall": 0.0, "missing_evidence_markers": ["marker"]},
        "fused": {"evidence_recall": 1.0},
        "pre_rerank": {"evidence_recall": 1.0},
        "reranked": {"evidence_recall": 0.0},
        "evidence_preservation_pool": {"evidence_recall": 1.0},
        "final": {"evidence_recall": 0.0},
    }
    final_score = {"passed": False, "expected_outcome": "answer", "failure_mode": "expected_evidence_missing"}

    assert module.classify_stage_loss(stage_scores, final_score) == "evidence_preservation_evidence_dropped"


def test_root_cause_audit_counts_document_first_as_source_pool() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")

    stage_hits = {stage: None for stage in module.PIPELINE_STAGES}
    stage_hits["expected_document_chunks"] = {"rank": 1}
    stage_hits["document_first_evidence"] = {"rank": 1}

    assert module.classify_marker_root_cause(stage_hits) == "fusion_dropped_source_hit"


def test_root_cause_audit_classifies_document_first_final_drop() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")

    stage_hits = {stage: None for stage in module.PIPELINE_STAGES}
    stage_hits["expected_document_chunks"] = {"rank": 1}
    stage_hits["document_first_evidence"] = {"rank": 1}
    stage_hits["fused"] = {"rank": 12}
    stage_hits["pre_rerank"] = {"rank": 12}
    stage_hits["reranked"] = {"rank": 8}

    assert module.classify_marker_root_cause(stage_hits) == "final_top_k_truncated_evidence"


def test_root_cause_audit_reports_query_evidence_overlap() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")

    overlap = module.query_marker_overlap(
        "客户手机号数据导出处理时限是多少",
        {"label": "客户手机号处理时限", "aliases": []},
        {"preview": "客户手机号数据导出审批人是信息安全负责人，处理时限为 2 个工作日。"},
    )
    no_overlap = module.query_marker_overlap(
        "客户手机号数据导出处理时限是多少",
        {"label": "仲裁法第二十一条", "aliases": []},
        {"preview": "当事人申请仲裁应当有仲裁协议。"},
    )

    assert overlap["overlap_ratio"] > 0
    assert no_overlap["overlap_ratio"] == 0


def test_root_cause_audit_reports_document_local_proximity_for_source_miss() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")
    document_id = uuid4()

    def stage_item(chunk_index: int, content: str) -> module.StageItem:
        return module.StageItem(
            chunk_id=uuid4(),
            document_id=document_id,
            document_title="enterprise:policy:数据安全管理办法",
            chunk_index=chunk_index,
            content=content,
        )

    stages = {stage: [] for stage in module.PIPELINE_STAGES}
    stages["expected_document_chunks"] = [stage_item(10, "目标证据：重要数据出境需要开展安全评估。")]
    stages["lexical"] = [stage_item(6, "重要数据处理活动应当建立分类分级保护制度。")]
    marker = {
        "label": "重要数据出境需要开展安全评估",
        "aliases": ["重要数据出境需要开展安全评估"],
        "document_title": "enterprise:policy:数据安全管理办法",
    }

    located = module.locate_marker(marker, stages, query="重要数据出境需要什么手续", match_mode="strict")
    summary = module.summarize(
        [
            {
                "case_root_cause": located["root_cause"],
                "diagnostics_stage_loss": "source_pool_evidence_missing",
                "markers": [located],
            }
        ]
    )

    proximity = located["document_local_proximity"]
    missed_summary = summary["source_candidate_generation_missed_proximity"]
    assert located["root_cause"] == "source_candidate_generation_missed"
    assert proximity["evidence_chunk_index"] == 10
    assert proximity["source_nearest"]["stage"] == "lexical"
    assert proximity["source_nearest"]["distance"] == 4
    assert proximity["source_within_windows"]["within_5"] is True
    assert missed_summary["within_window_counts"]["within_5"] == 1
    assert missed_summary["distance_median"] == 4


def test_root_cause_audit_summarizes_semantic_rerank_pool_visibility() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")

    summary = module.summarize(
        [
            {
                "case_root_cause": "only_recovered_by_in_document_expansion",
                "diagnostics_stage_loss": "source_pool_evidence_missing",
                "markers": [
                    {
                        "root_cause": "only_recovered_by_in_document_expansion",
                        "stage_ranks": {"fused": 120, "expansion": 3, "semantic_rerank_pool": 11},
                    },
                    {
                        "root_cause": "only_recovered_by_in_document_expansion",
                        "stage_ranks": {"fused": 150, "expansion": 8, "semantic_rerank_pool": None},
                    },
                    {
                        "root_cause": "source_candidate_generation_missed",
                        "stage_ranks": {"fused": None, "semantic_rerank_pool": None},
                    },
                ],
            }
        ]
    )

    visibility = summary["semantic_rerank_pool_visibility"]
    assert visibility["marker_count"] == 3
    assert visibility["visible_marker_count"] == 1
    assert visibility["fused_available_marker_count"] == 2
    assert visibility["fused_available_visible_count"] == 1
    assert visibility["expansion_recovered_marker_count"] == 2
    assert visibility["expansion_recovered_visible_count"] == 1
    assert visibility["visible_by_root_cause"] == {"only_recovered_by_in_document_expansion": 1}


def test_root_cause_audit_summarizes_semantic_rerank_pool_oracle_cases() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")

    summary = module.summarize(
        [
            {
                "case_name": "already",
                "case_root_cause": "passed",
                "diagnostics_stage_loss": "passed",
                "markers": [{"root_cause": "final_hit", "stage_ranks": {"final": 1}}],
            },
            {
                "case_name": "visible",
                "case_root_cause": "final_top_k_truncated_evidence",
                "diagnostics_stage_loss": "final_selection_evidence_dropped",
                "markers": [
                    {"root_cause": "final_hit", "stage_ranks": {"final": 1}},
                    {
                        "root_cause": "final_top_k_truncated_evidence",
                        "stage_ranks": {"fused": 24, "semantic_rerank_pool": 6, "final": None},
                    },
                ],
            },
            {
                "case_name": "not-generated",
                "case_root_cause": "source_candidate_generation_missed",
                "diagnostics_stage_loss": "source_pool_evidence_missing",
                "markers": [
                    {
                        "root_cause": "source_candidate_generation_missed",
                        "stage_ranks": {"fused": None, "semantic_rerank_pool": None, "final": None},
                    }
                ],
            },
            {
                "case_name": "not-visible",
                "case_root_cause": "rerank_truncated_evidence",
                "diagnostics_stage_loss": "rerank_evidence_dropped",
                "markers": [
                    {
                        "root_cause": "rerank_truncated_evidence",
                        "stage_ranks": {"fused": 84, "semantic_rerank_pool": None, "final": None},
                    }
                ],
            },
        ]
    )

    oracle = summary["semantic_rerank_pool_oracle"]
    assert oracle["classification_counts"] == {
        "already_final": 1,
        "semantic_pool_oracle_possible": 1,
        "blocked_by_candidate_generation": 1,
        "blocked_by_semantic_pool_visibility": 1,
    }
    assert oracle["oracle_rescuable_case_count"] == 1
    assert oracle["oracle_rescuable_case_names"] == ["visible"]


def test_root_cause_audit_summarizes_document_neighbor_candidate_oracle() -> None:
    module = _load_script_module("audit_retrieval_root_causes.py")

    summary = module.summarize(
        [
            {
                "case_name": "near",
                "case_root_cause": "source_candidate_generation_missed",
                "diagnostics_stage_loss": "source_pool_evidence_missing",
                "markers": [
                    {
                        "root_cause": "final_hit",
                        "stage_ranks": {"fused": 1, "final": 1},
                        "document_local_proximity": {"available": True},
                    },
                    {
                        "root_cause": "source_candidate_generation_missed",
                        "stage_ranks": {"fused": None, "final": None},
                        "document_local_proximity": {
                            "available": True,
                            "source_nearest": {"distance": 6, "stage": "lexical"},
                        },
                    },
                ],
            },
            {
                "case_name": "far",
                "case_root_cause": "source_candidate_generation_missed",
                "diagnostics_stage_loss": "source_pool_evidence_missing",
                "markers": [
                    {
                        "root_cause": "source_candidate_generation_missed",
                        "stage_ranks": {"fused": None, "final": None},
                        "document_local_proximity": {
                            "available": True,
                            "source_nearest": {"distance": 80, "stage": "lexical"},
                        },
                    }
                ],
            },
        ]
    )

    oracle = summary["document_neighbor_candidate_oracle"]["windows"]
    assert oracle["within_5"]["candidate_pool_oracle_possible_case_count"] == 0
    assert oracle["within_10"]["candidate_pool_oracle_possible_case_count"] == 1
    assert oracle["within_10"]["candidate_pool_oracle_possible_case_names"] == ["near"]
    assert oracle["within_50"]["candidate_pool_oracle_possible_case_count"] == 1
    assert oracle["within_10"]["not_fused_marker_neighbor_visible_count"] == 1
