"""Roles, document groups and who may read what.

Every document belongs to exactly one group, taken from the folder of its PDF
(data/raw/pdf_with_noise/<group>/). The roles in access_policy.json map to the groups
that a user with this role may read.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GROUPS = ("A", "B", "Geteilt")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf_with_noise"
OCR_DIR = PROJECT_ROOT / "data" / "preprocessed"
POLICY_PATH = Path(__file__).with_name("access_policy.json")

CHUNK_FILE_STEM = "synthetic_legal_counsel_documents_chunks"


def chunk_file_name(group: str) -> str:
    """Name of the chunk file (and of the RagFile in the corpus) of one document group."""
    return f"{CHUNK_FILE_STEM}_{group}.jsonl"


class UnknownRole(PermissionError):
    pass


@dataclass(frozen=True)
class Access:
    """The document groups one user may read and the id that owns their uploads."""

    user_id: str
    groups: frozenset[str]

    def can_read(self, group: str | None) -> bool:
        return group in self.groups


# For evaluations and scripts. Never build this from a request.
SYSTEM_ACCESS = Access("system", frozenset(GROUPS))


@lru_cache(maxsize=1)
def load_policy() -> dict[str, frozenset[str]]:
    raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy = {role: frozenset(groups) for role, groups in raw.items()}
    unknown = {g for groups in policy.values() for g in groups} - set(GROUPS)
    if unknown:
        raise ValueError(f"{POLICY_PATH.name} names unknown groups: {sorted(unknown)}")
    return policy


def access_for_roles(user_id: str, roles: Iterable[str]) -> Access:
    """Union of the groups of all known roles, unknown roles grant nothing."""
    policy = load_policy()
    groups = frozenset().union(*(policy.get(role, frozenset()) for role in roles))
    if not groups:
        raise UnknownRole(f"No known role in {list(roles)}")
    return Access(user_id, groups)


@lru_cache(maxsize=4)
def _document_groups(ocr_dir: Path) -> dict[str, str]:
    groups = {}
    for path in sorted(ocr_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if group := record.get("access_group"):
            groups[record["document_id"]] = group
    return groups


def document_groups(ocr_dir: Path = OCR_DIR) -> dict[str, str]:
    """document_id to group, read from the OCR results. Documents without a group are unknown."""
    return _document_groups(ocr_dir)
