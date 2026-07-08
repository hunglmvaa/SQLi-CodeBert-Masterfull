#!/usr/bin/env python3
"""Verify files against MANIFEST_SHA256.json."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for rel_path, expected in sorted(manifest.items()):
        path = ROOT / rel_path
        if not path.exists():
            failures.append((rel_path, "missing", expected))
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append((rel_path, actual, expected))

    if failures:
        print("Checksum verification failed:")
        for rel_path, actual, expected in failures:
            print(f"- {rel_path}: actual={actual} expected={expected}")
        return 1

    print(f"Checksum verification passed for {len(manifest)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

