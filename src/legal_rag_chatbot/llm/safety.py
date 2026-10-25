"""Protection against prompt injection in text that is passed to the model as data."""

import logging
import re

logger = logging.getLogger(__name__)

# Tags that mark data in the prompts (see prompts/v2/security.md).
DOCUMENTS_TAG = "dokumentauszuege"
HISTORY_TAG = "verlauf"

_DELIMITER = re.compile(
    rf"<\s*/?\s*(?:{DOCUMENTS_TAG}|{HISTORY_TAG})\s*>", re.IGNORECASE
)
# Only for logging, never for blocking: legal texts can legitimately quote such phrases.
_SUSPICIOUS = re.compile(
    r"(?:\b(?:ignorier\w*|vergiss|missachte\w*)\b.{0,40}\b(?:anweisung\w*|regeln|vorgaben|prompt)"
    r"|\b(?:ignore|disregard|forget)\b.{0,40}\b(?:instructions?|rules|prompt)"
    r"|\b(?:system\s?prompt|systemanweisung)\b"
    r"|\b(?:du bist|you are) (?:jetzt|ab sofort|now)\b"
    r"|!\[[^\]]*\]\(https?://)",
    re.IGNORECASE | re.DOTALL,
)


def neutralize(text: str) -> str:
    """Make text unable to close or fake the data tags it is placed in."""
    return _DELIMITER.sub(lambda m: m.group().replace("<", "‹").replace(">", "›"), text)


def log_if_suspicious(text: str, origin: str) -> None:
    """Warn about injection-like phrases; the text itself is not logged (confidential)."""
    if _SUSPICIOUS.search(text):
        logger.warning("Possible prompt injection in %s", origin)


def wrap(tag: str, body: str) -> str:
    return f"<{tag}>\n{body}\n</{tag}>"
