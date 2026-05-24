from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import select


@dataclass(frozen=True)
class ComparisonCase:
    label: str
    actor_email: str
    query: str
    expected_target_titles: tuple[str, ...] = ()
    expected_refusal: bool = False
    forbidden_document_titles: tuple[str, ...] = ()


CASES = [
    ComparisonCase(
        label="admin-l4",
        actor_email="admin@local.test",
        query="L4 高风险供应商的审批链路、复核周期和退出要求是什么？",
        expected_target_titles=("供应商准入、合同变更与临时采购协作规范",),
    ),
    ComparisonCase(
        label="admin-phone-export",
        actor_email="admin@local.test",
        query="包含客户手机号的数据导出由谁审批，处理时限和脱敏要求是什么？",
        expected_target_titles=("客户数据导出与临时权限管理办法", "运营审批与客户响应规范"),
    ),
    ComparisonCase(
        label="admin-data-processing",
        actor_email="admin@local.test",
        query="数据处理服务验收时需要哪些材料，验收人是谁，资料保留多久？",
        expected_target_titles=("供应商准入、合同变更与临时采购协作规范",),
    ),
    ComparisonCase(
        label="viewer-security-exception-keywords",
        actor_email="viewer@local.test",
        query="安全例外登记 补偿控制 审批人 到期时间",
        forbidden_document_titles=("安全例外登记",),
    ),
    ComparisonCase(
        label="admin-security-exception-controls",
        actor_email="admin@local.test",
        query="《安全例外登记》里对补偿控制有什么要求？",
        expected_target_titles=("安全例外登记",),
    ),
    ComparisonCase(
        label="manager-security-exception-deny",
        actor_email="manager@local.test",
        query="《安全例外登记》里对补偿控制有什么要求？",
        expected_refusal=True,
        forbidden_document_titles=("安全例外登记",),
    ),
    ComparisonCase(
        label="manager-platform-fields",
        actor_email="manager@local.test",
        query="平台发布手册要求发布工单至少写明哪些信息？",
        expected_target_titles=("平台发布手册",),
    ),
    ComparisonCase(
        label="viewer-platform-fields-deny",
        actor_email="viewer@local.test",
        query="《平台发布手册》要求发布工单至少写明哪些信息？",
        expected_refusal=True,
        forbidden_document_titles=("平台发布手册",),
    ),
]


PROFILES = {
    "heuristic": {
        "rerank_provider": "heuristic",
        "rerank_model": None,
        "qwen_rerank_model": None,
    },
    "llm": {
        "rerank_provider": "llm",
        "rerank_model": "deepseek-v4-flash",
        "qwen_rerank_model": None,
    },
    "qwen": {
        "rerank_provider": "qwen",
        "rerank_model": None,
        "qwen_rerank_model": "qwen3-rerank",
    },
}


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "backend" / "data" / "eval_outputs"


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0, min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1))))
    return float(ordered[rank])


def summarize_profile(profile_result: dict[str, Any]) -> dict[str, Any]:
    results = profile_result["results"]
    total_latencies = [item["total_elapsed_ms"] for item in results if item.get("total_elapsed_ms") is not None]
    rerank_latencies = [item["rerank_latency_ms"] for item in results if item.get("rerank_latency_ms") is not None]
    fallback_count = sum(1 for item in results if item.get("fallback"))
    permission_leak_count = sum(1 for item in results if item.get("permission_leak"))
    target_cases = [item for item in results if item.get("expected_target_titles")]
    target_hit_count = sum(1 for item in target_cases if item.get("target_hit"))
    target_hit_rate = (target_hit_count / len(target_cases)) if target_cases else None
    return {
        "profile": profile_result["profile"],
        "effective_provider": profile_result.get("effective_provider"),
        "avg_total_latency_ms": round(mean(total_latencies), 1) if total_latencies else None,
        "p50_total_latency_ms": round(_percentile(total_latencies, 50), 1) if total_latencies else None,
        "p95_total_latency_ms": round(_percentile(total_latencies, 95), 1) if total_latencies else None,
        "avg_rerank_latency_ms": round(mean(rerank_latencies), 1) if rerank_latencies else None,
        "fallback_count": fallback_count,
        "permission_leak_count": permission_leak_count,
        "target_hit_count": target_hit_count,
        "target_case_count": len(target_cases),
        "target_hit_rate": round(target_hit_rate, 3) if target_hit_rate is not None else None,
    }


def build_run_payload(profile_results: list[dict[str, Any]]) -> dict[str, Any]:
    generated_at = datetime.now().isoformat(timespec="seconds")
    return {
        "generated_at": generated_at,
        "case_count": len(CASES),
        "profiles": profile_results,
        "summary": [summarize_profile(item) for item in profile_results],
    }


def render_markdown_summary(run_payload: dict[str, Any]) -> str:
    lines = [
        "# Rerank Provider Comparison",
        "",
        f"- Generated at: `{run_payload['generated_at']}`",
        f"- Case count: `{run_payload['case_count']}`",
        "",
        "## Summary",
        "",
        "| Profile | Avg Total (ms) | P50 Total (ms) | P95 Total (ms) | Avg Rerank (ms) | Fallbacks | Permission Leaks | Target Hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in run_payload["summary"]:
        target_display = (
            f"{item['target_hit_count']}/{item['target_case_count']} ({item['target_hit_rate']:.1%})"
            if item["target_hit_rate"] is not None
            else "n/a"
        )
        lines.append(
            f"| `{item['profile']}` | {item['avg_total_latency_ms']} | {item['p50_total_latency_ms']} | "
            f"{item['p95_total_latency_ms']} | {item['avg_rerank_latency_ms']} | {item['fallback_count']} | "
            f"{item['permission_leak_count']} | {target_display} |"
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Profile | Case | Actor | Strategy | Total (ms) | Rerank (ms) | Fallback | Leak | Target Hit | Top-3 |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for profile in run_payload["profiles"]:
        for result in profile["results"]:
            top_titles = ", ".join(
                f"{item['document_title']}#{item['chunk_index']}" for item in result.get("matched_chunks", [])[:3]
            )
            target_display = "yes" if result.get("target_hit") else ("n/a" if not result.get("expected_target_titles") else "no")
            lines.append(
                f"| `{profile['profile']}` | `{result['label']}` | `{result['actor_email']}` | "
                f"`{result.get('rerank_strategy')}` | {result.get('total_elapsed_ms')} | "
                f"{result.get('rerank_latency_ms')} | {result.get('fallback')} | {result.get('permission_leak')} | "
                f"{target_display} | {top_titles} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(run_payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp_slug()
    json_path = output_dir / f"rerank-provider-compare-{timestamp}.json"
    md_path = output_dir / f"rerank-provider-compare-{timestamp}.md"
    json_path.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_summary(run_payload), encoding="utf-8")
    return json_path, md_path


def _apply_profile(profile_name: str) -> None:
    profile = PROFILES[profile_name]
    os.environ["RERANK_PROVIDER"] = profile["rerank_provider"] or ""
    if profile["rerank_model"]:
        os.environ["RERANK_MODEL"] = profile["rerank_model"]
    else:
        os.environ.pop("RERANK_MODEL", None)
    if profile["qwen_rerank_model"]:
        os.environ["QWEN_RERANK_MODEL"] = profile["qwen_rerank_model"]
    else:
        os.environ.pop("QWEN_RERANK_MODEL", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rerank providers on a shared query set.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["heuristic", "llm", "qwen"],
        choices=sorted(PROFILES.keys()),
        help="Rerank provider profiles to run.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory for timestamped JSON and Markdown artifacts.",
    )
    args = parser.parse_args()

    from app.core.config import get_settings
    from app.db.session import SessionLocal
    from app.models.user import User
    from app.schemas.search import SearchRequest
    from app.services.retrieval.service import RetrievalService

    comparison_results: list[dict[str, Any]] = []

    for profile_name in args.profiles:
        _apply_profile(profile_name)
        get_settings.cache_clear()
        settings = get_settings()

        session = SessionLocal()
        try:
            profile_results: list[dict[str, object]] = []
            for case in CASES:
                actor = session.scalar(select(User).where(User.email == case.actor_email))
                if actor is None:
                    profile_results.append(
                        {
                            "label": case.label,
                            "actor_email": case.actor_email,
                            "query": case.query,
                            "expected_target_titles": list(case.expected_target_titles),
                            "expected_refusal": case.expected_refusal,
                            "permission_leak": False,
                            "error": "actor_not_found",
                        }
                    )
                    continue

                service = RetrievalService(session)
                response = service.search(actor, SearchRequest(query=case.query, top_k=args.top_k))
                forbidden_visible = [
                    item.document_title
                    for item in response.matched_chunks
                    if item.document_title in case.forbidden_document_titles
                ]
                permission_leak = bool(forbidden_visible)
                target_hit = any(
                    item.document_title in case.expected_target_titles for item in response.matched_chunks
                ) if case.expected_target_titles else None
                profile_results.append(
                    {
                        "label": case.label,
                        "actor_email": case.actor_email,
                        "query": case.query,
                        "expected_target_titles": list(case.expected_target_titles),
                        "expected_refusal": case.expected_refusal,
                        "rerank_strategy": response.debug.rerank_strategy,
                        "latency_breakdown_ms": {
                            "query_rewrite": response.debug.query_rewrite_latency_ms,
                            "lexical_retrieval": response.debug.lexical_retrieval_latency_ms,
                            "vector_embedding": response.debug.vector_embedding_latency_ms,
                            "vector_retrieval": response.debug.vector_retrieval_latency_ms,
                            "fusion": response.debug.fusion_latency_ms,
                            "rerank": response.debug.rerank_latency_ms,
                        },
                        "query_rewrite_latency_ms": response.debug.query_rewrite_latency_ms,
                        "lexical_retrieval_latency_ms": response.debug.lexical_retrieval_latency_ms,
                        "vector_embedding_latency_ms": response.debug.vector_embedding_latency_ms,
                        "vector_retrieval_latency_ms": response.debug.vector_retrieval_latency_ms,
                        "fusion_latency_ms": response.debug.fusion_latency_ms,
                        "rerank_latency_ms": response.debug.rerank_latency_ms,
                        "total_elapsed_ms": response.debug.search_total_latency_ms,
                        "fallback": "fallback" in response.debug.rerank_strategy,
                        "permission_isolation_ok": not permission_leak,
                        "permission_leak": permission_leak,
                        "target_hit": target_hit,
                        "forbidden_visible": forbidden_visible,
                        "matched_chunks": [
                            {
                                "document_title": item.document_title,
                                "chunk_index": item.chunk_index,
                                "fused": item.score.fused,
                                "rerank": item.score.rerank,
                            }
                            for item in response.matched_chunks
                        ],
                    }
                )
            comparison_results.append(
                {
                    "profile": profile_name,
                    "effective_provider": settings.rerank_provider,
                    "effective_rerank_model": settings.effective_rerank_model,
                    "effective_qwen_rerank_model": settings.effective_qwen_rerank_model,
                    "results": profile_results,
                }
            )
        finally:
            session.close()

    run_payload = build_run_payload(comparison_results)
    json_path, md_path = write_outputs(run_payload, Path(args.output_dir))
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "summary": run_payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
