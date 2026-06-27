from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "run_benchmark_eval.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_eval", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classify_failure_prefers_pipeline_diagnosis() -> None:
    module = _load_module()
    details = {
        "pipeline_diagnosis": {
            "stage": "candidate_selection",
            "reason_code": "expected_evidence_not_selected",
            "reason_label": "召回到了相关候选，但最终没有选中期望证据",
        },
        "metric_breakdown": {
            "retrieval": {"score": 0.9},
            "citation": {"score": 0.2},
            "faithfulness": {"score": 0.3},
            "permission_isolation": {"passed": True},
        },
    }
    result = {
        "overall_pass": False,
        "permission_isolation_correct": True,
    }

    assert module.classify_failure(details, result) == "expected_evidence_not_selected"


def test_build_failure_summary_counts_pipeline_stages() -> None:
    module = _load_module()
    report = {
        "results": [
            {
                "overall_pass": False,
                "permission_isolation_correct": True,
                "details_json": {
                    "case_annotations": {"expected_outcome": "answer"},
                    "human_review": {"reason": "selected evidence incomplete"},
                    "pipeline_diagnosis": {
                        "stage": "citation_coverage",
                        "reason_code": "selected_citations_missing_required_facts",
                        "reason_label": "已选引用未覆盖所需事实",
                    },
                    "metric_breakdown": {
                        "retrieval": {"score": 0.8},
                        "citation": {"score": 0.4},
                        "faithfulness": {"score": 0.6},
                        "permission_isolation": {"passed": True},
                    },
                },
            }
        ]
    }

    summary = module.build_failure_summary(report)
    failures = module.extract_failure_cases(report)

    assert summary["failure_stage_counts"] == {"citation_coverage": 1}
    assert summary["failure_mode_counts"] == {"selected_citations_missing_required_facts": 1}
    assert failures[0]["pipeline_diagnosis"]["stage"] == "citation_coverage"
