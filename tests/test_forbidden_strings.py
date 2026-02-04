"""
Forbidden-strings test: ensures the demo repo stays on-narrative.

If any shipped source or README contains tokens from the excluded scope,
this test fails. This prevents Frankenstein drift.
"""

import pathlib
import re
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Tokens that MUST NOT appear in the shipped demo codebase
# (case-insensitive match on whole words or substrings)
FORBIDDEN = [
    r"\bcrp\b",
    r"credit.risk.premium",
    r"\boverlay\b",
    r"\bstress_flag\b",
    r"\btail_stress_flag\b",
    r"\bregime.vector\b",
    r"\bDESK_MEMO\b",
    r"\balpha.claim\b",
    r"\bdrawdown.improvement\b",
]

SEARCH_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "configs",
]

SEARCH_FILES = [
    REPO_ROOT / "README.md",
]


def _all_files():
    """Yield all .py, .yaml, .md files under search dirs."""
    for d in SEARCH_DIRS:
        if d.exists():
            for f in d.rglob("*"):
                if f.suffix in (".py", ".yaml", ".yml", ".md"):
                    yield f
    for f in SEARCH_FILES:
        if f.exists():
            yield f


@pytest.mark.parametrize("pattern", FORBIDDEN)
def test_no_forbidden_token(pattern):
    """No shipped file contains a forbidden narrative token."""
    regex = re.compile(pattern, re.IGNORECASE)
    violations = []
    for fpath in _all_files():
        text = fpath.read_text(errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                violations.append(f"{fpath.relative_to(REPO_ROOT)}:{i}  {line.strip()[:80]}")
    assert violations == [], (
        f"Forbidden token /{pattern}/ found in:\n" + "\n".join(violations)
    )
