"""Shared utilities for Kuku ingestion, indexing, and retrieval.

Text cleaning is conservative: keep legal punctuation, clause numbers,
paragraphs, letter case and language-specific joiners intact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO


_REMOVED_FORMATTING = str.maketrans({
    "\u00ad": None,  # Soft hyphen, not an ordinary printed hyphen.
    "\ufeff": None,  # Byte-order mark embedded in extracted text.
    "\u200b": None,  # Zero-width space; retain ZWJ and ZWNJ.
})
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def sha256_file(path: str | Path) -> str:
    """Hash file bytes in bounded memory; missing/unreadable files raise."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: str) -> str:
    """Normalize extraction noise without rewriting legal content.

    Non-string input returns an empty string for backward compatibility.
    This is not an OCR repair function: it does not guess missing words,
    join hard-hyphenated words, lowercase text or remove repeated headings.
    """
    if not isinstance(value, str):
        return ""

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u2028", "\n")
    value = value.replace("\u2029", "\n\n").replace("\f", "\n\n")
    value = value.translate(_REMOVED_FORMATTING)
    value = _CONTROL_CHARACTERS.sub("", value)
    value = _HORIZONTAL_WHITESPACE.sub(" ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    # Normalize after removing formatting so repeated cleaning is idempotent.
    # NFC preserves superscripts and other potentially meaningful symbols.
    return unicodedata.normalize("NFC", value.strip())


def _atomic_write(path: str | Path, writer: Callable[[TextIO], None]) -> None:
    """Replace one destination only after serialization and file sync succeed.

    The temporary file shares the destination directory for atomic replacement.
    This does not make multiple index files a transaction or serialize writers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            # Cleanup must not hide the original serialization or I/O error.
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Stream dictionary records to UTF-8 JSONL without partial replacement."""
    def writer(handle: TextIO) -> None:
        for record_number, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise TypeError(
                    f"JSONL record {record_number} must be a dictionary"
                )
            # JSON has no NaN or Infinity values. Reject invalid index metadata.
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            handle.write("\n")

    _atomic_write(path, writer)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed")


def _parse_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"JSON number {value!r} is outside the finite float range")
    return number


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def iter_jsonl(
    path: str | Path, *, missing_ok: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield validated records with physical line numbers in error messages.

    Accept a UTF-8 BOM at the start of a file and ignore blank lines.
    Set missing_ok=False for required index files. A streaming consumer can
    receive earlier records before a later malformed record raises.
    """
    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except FileNotFoundError:
        if missing_ok:
            return
        raise

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    parse_float=_parse_json_float,
                    object_pairs_hook=_unique_json_object,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            yield value


def read_jsonl(
    path: str | Path, *, missing_ok: bool = True,
) -> list[dict[str, Any]]:
    """Read all JSONL records; preserve the original missing-file default."""
    return list(iter_jsonl(path, missing_ok=missing_ok))


def dump_json(path: str | Path, value: Any) -> None:
    """Write human-readable, finite UTF-8 JSON through atomic replacement."""
    def writer(handle: TextIO) -> None:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    _atomic_write(path, writer)


def slugify(value: str) -> str:
    """Make an ASCII filename component, not a unique document identifier.

    Preserve the original 'source' fallback. Use a document hash alongside
    slugs where uniqueness matters, especially for non-Latin filenames.
    """
    if not isinstance(value, str):
        raise TypeError("slugify expects a string")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-") or "source"
    if slug in _WINDOWS_RESERVED_NAMES:
        slug = f"source-{slug}"
    return slug
