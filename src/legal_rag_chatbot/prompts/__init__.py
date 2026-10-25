"""Versioned prompt sets. Each version is a directory of Markdown files next to this module.

The active version is fixed per program run via the PROMPT_VERSION environment variable.
To change a prompt, copy the latest version directory (e.g. v1 to v2), edit the copy and
select it. Old versions stay untouched, so old evaluation reports remain reproducible.
"""

import hashlib
import os
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
DEFAULT_VERSION = "v2"
VERSION = os.environ.get("PROMPT_VERSION", DEFAULT_VERSION)


def available_versions() -> list[str]:
    return sorted(
        p.name for p in PROMPTS_DIR.iterdir() if p.is_dir() and p.name[0] == "v"
    )


def _version_dir(version: str) -> Path:
    path = PROMPTS_DIR / version
    if not path.is_dir():
        raise ValueError(
            f"Unknown prompt version {version!r}, available: {available_versions()}"
        )
    return path


def get(name: str, version: str | None = None) -> str:
    """Text of one prompt without surrounding whitespace."""
    path = _version_dir(version or VERSION) / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def fingerprint(version: str | None = None) -> str:
    """Short hash over all prompt files of a version, to tell edited versions apart."""
    digest = hashlib.sha256()
    for path in sorted(_version_dir(version or VERSION).glob("*.md")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def info() -> dict[str, str]:
    """Version and hash of the active prompt set, for evaluation reports."""
    return {"prompt_version": VERSION, "prompt_hash": fingerprint()}
