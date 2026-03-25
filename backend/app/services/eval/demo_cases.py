from __future__ import annotations

DEMO_EVAL_CASES: list[dict] = [
    {
        "dataset_name": "demo_permission_eval",
        "case_name": "manager_can_find_platform_runbook",
        "description": "Manager should retrieve the platform runbook shared to the platform team.",
        "acting_user_email": "manager@local.test",
        "question": "What does the platform release checklist require?",
        "expected_document_titles": ["Platform Runbook"],
        "forbidden_document_titles": [],
        "expected_answer_keywords": ["release", "checklist"],
        "notes": "Requires a team-visible Platform Runbook document.",
    },
    {
        "dataset_name": "demo_permission_eval",
        "case_name": "viewer_cannot_see_platform_runbook",
        "description": "Viewer should not retrieve the platform runbook when it is only team-visible.",
        "acting_user_email": "viewer@local.test",
        "question": "What does the platform release checklist require?",
        "expected_document_titles": [],
        "forbidden_document_titles": ["Platform Runbook"],
        "expected_answer_keywords": [],
        "notes": "Permission isolation case. Expected answer is cautious or insufficient.",
    },
    {
        "dataset_name": "demo_permission_eval",
        "case_name": "viewer_can_find_public_handbook",
        "description": "Viewer should retrieve the public handbook for public information.",
        "acting_user_email": "viewer@local.test",
        "question": "What does the company handbook say about holiday schedule?",
        "expected_document_titles": ["Public Handbook"],
        "forbidden_document_titles": ["Platform Runbook"],
        "expected_answer_keywords": ["holiday", "schedule"],
        "notes": "Requires a public Public Handbook document.",
    },
]
