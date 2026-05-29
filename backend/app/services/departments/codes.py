from __future__ import annotations

from secrets import choice

ORG_CODE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ORG_CODE_ROOT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STABLE_CODE_LENGTH = 5
MAX_ORG_CODE_LENGTH = 64


class DepartmentCodeSpaceExhausted(ValueError):
    pass


def generate_stable_code(existing_codes: set[str]) -> str:
    for _ in range(1024):
        code = choice(ORG_CODE_ROOT_LETTERS) + "".join(choice(ORG_CODE_CHARS) for _ in range(STABLE_CODE_LENGTH - 1))
        if code not in existing_codes:
            return code
    raise DepartmentCodeSpaceExhausted("稳定编号空间暂时无法分配")


def generate_root_org_code(existing_org_codes: set[str]) -> str:
    for letter in ORG_CODE_ROOT_LETTERS:
        if not any(code.startswith(letter) for code in existing_org_codes):
            return f"{letter}{choice(ORG_CODE_CHARS)}"

    candidates = [
        f"{letter}{suffix}"
        for letter in ORG_CODE_ROOT_LETTERS
        for suffix in ORG_CODE_CHARS
        if f"{letter}{suffix}" not in existing_org_codes
    ]
    if not candidates:
        raise DepartmentCodeSpaceExhausted("一级部门编号空间已满")
    return choice(candidates)


def generate_child_org_code(parent_org_code: str, sibling_org_codes: set[str]) -> str:
    if len(parent_org_code) + 1 > MAX_ORG_CODE_LENGTH:
        raise DepartmentCodeSpaceExhausted("部门编号层级超过上限")

    candidates = [f"{parent_org_code}{char}" for char in ORG_CODE_CHARS]
    available = [code for code in candidates if code not in sibling_org_codes]
    if not available:
        raise DepartmentCodeSpaceExhausted("同级部门数量已达到当前编号规则上限")
    return choice(available)


def build_org_code_path(parent_org_code_path: str | None, org_code: str) -> str:
    return f"{parent_org_code_path}/{org_code}" if parent_org_code_path else f"/{org_code}"
