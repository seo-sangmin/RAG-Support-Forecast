from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class JsonCache:
    """Content-hash-keyed JSON cache backed by one file per entry."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> Any | None:
        path = self._path(_key(payload))
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put(self, payload: dict[str, Any], value: Any) -> None:
        path = self._path(_key(payload))
        path.write_text(json.dumps(value, default=str))
