from __future__ import annotations

import hashlib
import json
from typing import Any

HASH_ALGORITHM = "sha256"
HASH_VERSION = "v2"


def normalize_content(content: str) -> str:
    """Canonicalize line endings without removing meaningful whitespace."""
    return content.replace("\r\n", "\n").replace("\r", "\n")


def hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{HASH_ALGORITHM}:{hashlib.sha256(encoded).hexdigest()}"
