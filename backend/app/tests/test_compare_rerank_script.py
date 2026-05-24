from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_compare_script_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "compare_rerank_providers.py"
    spec = importlib.util.spec_from_file_location("compare_rerank_providers", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_rerank_summary_and_output_files(tmp_path: Path) -> None:
    module = _load_compare_script_module()
    profile_results = [
        {
            "profile": "heuristic",
            "effective_provider": "heuristic",
            "results": [
                {
                    "label": "case-a",
                    "actor_email": "viewer@example.com",
                    "rerank_strategy": "heuristic-overlap",
                    "total_elapsed_ms": 1000,
                    "rerank_latency_ms": 20,
                    "fallback": False,
                    "permission_leak": False,
                    "expected_target_titles": ["Doc A"],
                    "target_hit": True,
                    "matched_chunks": [{"document_title": "Doc A", "chunk_index": 0}],
                },
                {
                    "label": "case-b",
                    "actor_email": "viewer@example.com",
                    "rerank_strategy": "heuristic-overlap",
                    "total_elapsed_ms": 1400,
                    "rerank_latency_ms": 30,
                    "fallback": False,
                    "permission_leak": False,
                    "expected_target_titles": [],
                    "target_hit": None,
                    "matched_chunks": [{"document_title": "Doc B", "chunk_index": 1}],
                },
            ],
        },
        {
            "profile": "qwen",
            "effective_provider": "qwen",
            "results": [
                {
                    "label": "case-a",
                    "actor_email": "viewer@example.com",
                    "rerank_strategy": "qwen-rerank",
                    "total_elapsed_ms": 1800,
                    "rerank_latency_ms": 300,
                    "fallback": False,
                    "permission_leak": False,
                    "expected_target_titles": ["Doc A"],
                    "target_hit": False,
                    "matched_chunks": [{"document_title": "Doc C", "chunk_index": 2}],
                },
                {
                    "label": "case-b",
                    "actor_email": "manager@example.com",
                    "rerank_strategy": "qwen-rerank-fallback-heuristic",
                    "total_elapsed_ms": 2200,
                    "rerank_latency_ms": 8100,
                    "fallback": True,
                    "permission_leak": True,
                    "expected_target_titles": [],
                    "target_hit": None,
                    "matched_chunks": [{"document_title": "Secret", "chunk_index": 3}],
                },
            ],
        },
    ]

    run_payload = module.build_run_payload(profile_results)
    summary_by_profile = {item["profile"]: item for item in run_payload["summary"]}

    heuristic_summary = summary_by_profile["heuristic"]
    assert heuristic_summary["avg_total_latency_ms"] == 1200
    assert heuristic_summary["avg_rerank_latency_ms"] == 25
    assert heuristic_summary["fallback_count"] == 0
    assert heuristic_summary["permission_leak_count"] == 0
    assert heuristic_summary["target_hit_count"] == 1
    assert heuristic_summary["target_case_count"] == 1
    assert heuristic_summary["target_hit_rate"] == 1.0

    qwen_summary = summary_by_profile["qwen"]
    assert qwen_summary["fallback_count"] == 1
    assert qwen_summary["permission_leak_count"] == 1
    assert qwen_summary["target_hit_count"] == 0
    assert qwen_summary["target_case_count"] == 1
    assert qwen_summary["target_hit_rate"] == 0.0

    markdown = module.render_markdown_summary(run_payload)
    assert "# Rerank Provider Comparison" in markdown
    assert "| `heuristic` | 1200" in markdown
    assert "| `qwen` | 2000" in markdown
    assert "1/1 (100.0%)" in markdown
    assert "0/1 (0.0%)" in markdown

    json_path, md_path = module.write_outputs(run_payload, tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.read_text(encoding="utf-8")
    assert md_path.read_text(encoding="utf-8").startswith("# Rerank Provider Comparison")
