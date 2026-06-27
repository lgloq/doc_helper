from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "build_benchmark_layer_manifest.py"
    spec = importlib.util.spec_from_file_location("build_benchmark_layer_manifest", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_layer_catalog_uses_official_four_layer_scheme() -> None:
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[3]

    report = module.build_layer_catalog(
        manifest_path=repo_root / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_case_manifest_strict_evidence_verified.json",
        ingestion_quality_path=repo_root / "backend" / "data" / "eval_outputs" / "zh-enterprise-v1-verified234-final-ingestion-quality-local.json",
    )

    assert report["layer_order"] == ["smoke", "full", "hard", "latency"]
    assert report["benchmark_positioning"]["scope"] == "Chinese enterprise-document RAG"
    assert report["benchmark_positioning"]["evaluation_corpus_document_count"] == 102
    assert report["benchmark_positioning"]["evaluation_corpus_chunk_count"] == 17557

    smoke = report["layers"]["smoke"]
    full = report["layers"]["full"]
    hard = report["layers"]["hard"]
    latency = report["layers"]["latency"]

    assert smoke["case_count"] == 24
    assert smoke["layer_referenced_document_count"] == 19
    assert smoke["layer_referenced_chunk_count"] == 1358
    assert smoke["case_type_counts"] == {
        "multi_evidence_cross_document": 4,
        "multi_evidence_same_document": 4,
        "permission": 4,
        "single_fact": 4,
        "table_structured": 4,
        "version_temporal": 4,
    }

    assert full["case_count"] == 234
    assert full["layer_referenced_document_count"] == 85
    assert full["layer_referenced_chunk_count"] == 15706

    assert hard["case_count"] == 116
    assert hard["case_type_counts"] == {
        "multi_evidence_cross_document": 33,
        "permission": 22,
        "table_structured": 37,
        "version_temporal": 24,
    }
    assert hard["layer_referenced_document_count"] == 50
    assert hard["layer_referenced_chunk_count"] == 13278

    assert latency["case_count"] == 24
    assert latency["layer_referenced_document_count"] == 12
    assert latency["layer_referenced_chunk_count"] == 6794
    assert latency["case_type_counts"] == smoke["case_type_counts"]


def test_emit_layer_manifests_keeps_full_document_corpus(tmp_path: Path) -> None:
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[3]

    report = module.build_layer_catalog(
        manifest_path=repo_root / "backend" / "data" / "benchmark_raw" / "zh_enterprise" / "v1_case_manifest_strict_evidence_verified.json",
        ingestion_quality_path=repo_root / "backend" / "data" / "eval_outputs" / "zh-enterprise-v1-verified234-final-ingestion-quality-local.json",
    )
    module.emit_layer_manifests(report, emit_dir=tmp_path)

    smoke_manifest = json.loads((tmp_path / "v1_case_manifest_strict_evidence_verified_smoke.json").read_text(encoding="utf-8"))
    hard_manifest = json.loads((tmp_path / "v1_case_manifest_strict_evidence_verified_hard.json").read_text(encoding="utf-8"))
    latency_manifest = json.loads((tmp_path / "v1_case_manifest_strict_evidence_verified_latency.json").read_text(encoding="utf-8"))

    assert len(smoke_manifest["documents"]) == 102
    assert len(smoke_manifest["cases"]) == report["layers"]["smoke"]["case_count"]
    assert smoke_manifest["layer_profile"]["manifest_mode"] == "case_subset_on_full_corpus"

    assert len(hard_manifest["documents"]) == 102
    assert len(hard_manifest["cases"]) == report["layers"]["hard"]["case_count"]

    assert len(latency_manifest["documents"]) == 102
    assert len(latency_manifest["cases"]) == report["layers"]["latency"]["case_count"]
