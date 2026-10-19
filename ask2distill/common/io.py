"""Small file helpers: jsonl, parquet, atomic writes, md5.

Writes go to a temporary file in the destination directory and are renamed
into place, so a job killed at the wall clock never leaves a truncated output
that a later resume would mistake for a finished one.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def _tmp_in(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically (same-directory rename)."""
    path = Path(path)
    fd, tmp = _tmp_in(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def atomic_write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` atomically."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int | None = 1) -> Path:
    """Write ``obj`` as JSON (UTF-8, non-ASCII kept) atomically."""
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    """All records of a jsonl file, skipping blank lines."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    """Records of a jsonl file, one at a time."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> Path:
    """Write records as jsonl atomically, with ``ensure_ascii=False`` like the originals."""
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return atomic_write_text(path, text)


def count_lines(path: str | os.PathLike) -> int:
    """Number of lines in a text file (0 if it does not exist)."""
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def read_parquet(path: str | os.PathLike, columns: list[str] | None = None):
    """A parquet file as a pandas DataFrame."""
    import pandas as pd
    return pd.read_parquet(path, columns=columns)


def write_parquet(path: str | os.PathLike, df, index: bool = False) -> Path:
    """Write a DataFrame to parquet atomically."""
    path = Path(path)
    fd, tmp = _tmp_in(path)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=index)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def md5_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """Hex md5 of a file's bytes."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def md5_text(text: str) -> str:
    """Hex md5 of a UTF-8 string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()
