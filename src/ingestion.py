"""Extract PDF pages and TXT/FAQ documents with source provenance.

Run from the project root with: python -m src.ingestion
Place this file and utils.py in src/. No API key or GPU is required.
Scanned PDFs are flagged for review; this module does not perform OCR.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import pymupdf

from .utils import _atomic_write, clean_text, dump_json, sha256_file, write_jsonl


LOGGER = logging.getLogger(__name__)
_TITLE_END = re.compile(
    r"(?:\b(?:ACT|CODE|SANHITA|RULES|REGULATIONS|ORDERS?)\b.{0,80}"
    r"\b(?:18|19|20)\d{2}|\bCONSTITUTION OF INDIA)\s*[.:-]?$",
    re.IGNORECASE,
)
_NOISY_TITLE = re.compile(
    r"\b(?:ARRANGEMENT OF SECTIONS|TABLE OF CONTENTS|MINISTRY OF|"
    r"GOVERNMENT OF|GAZETTE OF|LEGISLATIVE DEPARTMENT|AMENDMENT ACTS|"
    r"WHEREAS|HEREBY|ENACTED|MEANS)\b", re.IGNORECASE,
)
# Match boundaries independently of Question/Answer validity. A malformed FAQ
# must not silently become part of the preceding FAQ's answer.
_FAQ_HEADER = re.compile(r"^[ \t]*FAQ[ \t]+(\d+)[ \t]*:?[ \t]*$", re.I | re.M)
_QA_PATTERN = re.compile(
    r"^[ \t]*Question:[ \t]*(?P<question>.*?)"
    r"^[ \t]*Answer:[ \t]*(?P<answer>.*)\Z", re.I | re.M | re.S,
)
_SOURCE_PATTERN = re.compile(r"^[ \t]*Source:[ \t]*(https?://\S+)[ \t]*$", re.I | re.M)
_TRAILING_SEPARATOR = re.compile(r"(?:\n[ \t]*[-=]{8,}[ \t]*)+\Z")
_SCRIPT_RUN = re.compile(r"[\u0900-\u097f\u0370-\u052f]+")
_DEVANAGARI = re.compile(r"[\u0904-\u0939\u0958-\u0961]")
_GREEK_OR_CYRILLIC = re.compile(r"[\u0370-\u052f]")
_META_FIELDS = (
    "document_title", "title_source", "title_verified", "document_type",
    "jurisdiction", "effective_date", "source_url",
)


def _candidate_title(text: str) -> str | None:
    """Find a title candidate, never proof of authenticity or current law."""
    lines = [re.sub(r"\s+", " ", line).strip(" -:.")
             for line in text.splitlines()[:100]]
    lines = [line for line in lines if line]
    for start in range(len(lines)):
        for width in (1, 2, 3):
            title = " ".join(lines[start:start + width])
            if not 12 <= len(title) <= 180 or len(title.split()) < 3:
                continue
            if _NOISY_TITLE.search(title) or not _TITLE_END.search(title):
                continue
            # Remove extraction noise before a clear title opening.
            opening = re.search(r"\bTHE\s+", title, re.I)
            if opening:
                title = title[opening.start():]
            return title
    # Guides do not necessarily name an Act. Only accept a short heading,
    # never a whole paragraph or a claim that the title has been verified.
    guide_heading = re.compile(
        r"\b(?:FAQs?|frequently asked questions|handbook|user manual|"
        r"guidelines|legal tool ?kit|an easy guide)\b", re.I,
    )
    for line in lines[:45]:
        if (12 <= len(line) <= 180 and len(line.split()) >= 3
                and guide_heading.search(line) and not _NOISY_TITLE.search(line)):
            return line
    return None


def _valid_source_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and not any(c.isspace() for c in value)
        )
    except ValueError:
        return False


def _validate_override(value: Any, relative_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid metadata override for {relative_path}: expected object")
    result = dict(value)
    for key in ("document_title", "document_type", "jurisdiction", "effective_date", "source_url"):
        item = result.get(key)
        if item is not None and not isinstance(item, str):
            raise ValueError(f"{relative_path}: metadata {key} must be a string or null")
        if isinstance(item, str):
            result[key] = item.strip() or None
    if "title_verified" in result and type(result["title_verified"]) is not bool:
        raise ValueError(f"{relative_path}: title_verified must be a boolean")
    if result.get("title_verified") and not result.get("document_title"):
        raise ValueError(f"{relative_path}: a verified title requires document_title")
    if result.get("source_url") and not _valid_source_url(result["source_url"]):
        raise ValueError(f"{relative_path}: source_url must be an HTTP(S) URL")
    return result


def _document_metadata(
    path: Path, relative_path: str, first_pages_text: str,
    overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Prefer human metadata, then a PDF title candidate, then the filename."""
    path = Path(path)
    override = _validate_override(overrides.get(relative_path, {}), relative_path)
    explicit_title = override.get("document_title")
    inferred_title = _candidate_title(first_pages_text) if path.suffix.lower() == ".pdf" else None
    if explicit_title:
        title, title_source = explicit_title, "manual_override"
    elif inferred_title:
        title, title_source = inferred_title, "pdf_text_candidate"
    else:
        title = path.stem.replace("_", " ").replace("-", " ")
        title_source = "filename_fallback"
    return {
        "document_title": title,
        "title_source": title_source,
        # Supplying a title does not prove that someone checked it.
        "title_verified": bool(explicit_title) and override.get("title_verified", False),
        "document_type": override.get("document_type") or (
            "legal_act" if relative_path.partition("/")[0].casefold() == "acts" else "guide"
        ),
        # Folder labels classify input only; they do not establish applicability.
        "jurisdiction": override.get("jurisdiction"),
        "effective_date": override.get("effective_date"),
        "source_url": override.get("source_url"),
    }


def make_record(
    path: Path, document_id: str, relative_path: str, text: str,
    source_type: str, page_number: int | None = None,
    faq_number: int | None = None, source_url: str | None = None,
    question: str | None = None, document_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the existing record schema and stable, content-based identifiers."""
    path = Path(path)
    info = document_meta or {}
    locator = (f"page_{page_number}" if page_number is not None else
               f"faq_{faq_number}" if faq_number is not None else "text_1")
    return {
        "record_id": f"{document_id}-{locator}",
        "document_id": document_id,
        "document_name": path.name,
        "document_title": info.get("document_title") or path.stem,
        "title_source": info.get("title_source", "filename_fallback"),
        "title_verified": bool(info.get("title_verified", False)),
        "document_type": info.get("document_type", "unknown"),
        "source_path": str(path.resolve()),
        "relative_source_path": relative_path,
        "source_type": source_type,
        # These are physical, one-based PDF pages, not printed page labels.
        "page_number": page_number, "page_start": page_number, "page_end": page_number,
        "faq_number": faq_number, "question": question,
        "section": f"FAQ {faq_number}" if faq_number is not None else None,
        "text": text,
        "source_url": info.get("source_url") or source_url,
        "jurisdiction": info.get("jurisdiction"),
        "effective_date": info.get("effective_date"),
        "char_count": len(text),
    }


def extract_pdf(
    path: Path, document_id: str, relative_path: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Extract each page once; isolate page failures without renumbering pages."""
    path = Path(path)
    warnings: list[str] = []
    pages: list[tuple[int, str]] = []
    empty_pages = short_pages = failed_pages = 0
    page_issues: list[dict[str, Any]] = []
    suspect_pages = 0
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDF: decrypt it before ingestion")
        total_pages = len(pdf)
        for index in range(total_pages):
            page_number = index + 1
            try:
                text = clean_text(pdf[index].get_text("text", sort=True))
            except Exception as exc:
                failed_pages += 1
                page_issues.append({"page_number": page_number, "status": "error"})
                warnings.append(f"Page {page_number}: {type(exc).__name__}: {exc}")
                continue
            if not text:
                empty_pages += 1
                page_issues.append({"page_number": page_number, "status": "empty"})
                warnings.append(f"Page {page_number}: no extractable text (possible scan/blank page)")
                continue
            if len(text) < 30:
                short_pages += 1
                warnings.append(f"Page {page_number}: very little extractable text")
            mixed_words = sum(
                bool(_DEVANAGARI.search(word) and _GREEK_OR_CYRILLIC.search(word))
                for word in _SCRIPT_RUN.findall(text)
            )
            if mixed_words >= 3:
                suspect_pages += 1
                page_issues.append({
                    "page_number": page_number, "status": "suspect_text_encoding",
                })
            pages.append((page_number, text))

    if suspect_pages:
        warnings.append(
            f"{suspect_pages}/{total_pages} pages contain words mixing Devanagari with Greek "
            "or Cyrillic characters; check the flagged pages against the PDF and use "
            "a corrected text layer or OCR if needed"
        )

    # Catch image-heavy PDFs whose only extractable content is page numbers
    # or cover credits. Do not claim OCR succeeded merely because records exist.
    useful_pages = [(number, text) for number, text in pages
                    if sum(character.isalpha() for character in text) >= 40]
    low_text_pages = total_pages - len(useful_pages) - failed_pages
    if total_pages >= 3 and low_text_pages / total_pages > 0.20:
        warnings.append(
            f"{low_text_pages}/{total_pages} pages have very little useful text; "
            "content may be missing or require OCR"
        )
    if pages and not useful_pages:
        warnings.append("No substantive page text found; page numbers/credits are not usable content")
        pages = []
    elif (total_pages >= 5 and len(useful_pages) / total_pages < 0.20
          and sum(len(text) for _, text in useful_pages) < 1500):
        warnings.append("Almost all page content is unreadable; OCR required before publication")
        pages = []

    opening_text = "\n".join(text[:3500] for number, text in pages if number <= 6)
    info = _document_metadata(path, relative_path, opening_text, overrides or {})
    if info["title_source"] == "filename_fallback":
        warnings.append("Document title could not be identified; provide a metadata override")
    if total_pages and empty_pages / total_pages > 0.20:
        warnings.append("More than 20% of pages have no text; inspect for scans requiring OCR")
    if not total_pages:
        warnings.append("PDF contains no pages")
    records = [make_record(path, document_id, relative_path, text, "pdf",
                           page_number=number, document_meta=info)
               for number, text in pages]
    return records, warnings, {
        "total_pages": total_pages, "empty_pages": empty_pages,
        "short_pages": short_pages, "failed_pages": failed_pages,
        "page_issues": page_issues, **info,
    }


def _source_url(text: str, warnings: list[str], label: str) -> str | None:
    urls = [match.group(1) for match in _SOURCE_PATTERN.finditer(text)]
    valid = list(dict.fromkeys(url for url in urls if _valid_source_url(url)))
    if len(valid) != len(set(urls)):
        warnings.append(f"{label}: invalid Source URL ignored")
    if len(valid) > 1:
        warnings.append(f"{label}: multiple Source URLs; first used as primary")
    return valid[0] if valid else None


def _missing_faq_ranges(numbers: set[int]) -> str:
    """Describe gaps without allocating a set spanning arbitrary FAQ numbers."""
    ordered = sorted(numbers)
    gaps: list[str] = []
    for left, right in zip(ordered, ordered[1:]):
        if right > left + 1:
            gaps.append(str(left + 1) if right == left + 2 else f"{left + 1}-{right - 1}")
    return ", ".join(gaps[:20]) + (", ..." if len(gaps) > 20 else "")


def extract_txt(
    path: Path, document_id: str, relative_path: str,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Split FAQs safely; retain malformed blocks and introductory text."""
    path = Path(path)
    warnings: list[str] = []
    payload = path.read_bytes()
    try:
        raw = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = payload.decode("utf-8-sig", errors="replace")
        warnings.append("Invalid UTF-8 replaced; review text encoding before indexing")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    info = _document_metadata(path, relative_path, "", overrides or {})
    records: list[dict[str, Any]] = []
    headers = list(_FAQ_HEADER.finditer(raw))
    preamble = raw[:headers[0].start()] if headers else raw
    default_url = _source_url(preamble, warnings, "Document")
    stats = {"total_pages": None, "empty_pages": None, "short_pages": None,
             "failed_pages": 0, "page_issues": [], **info}
    stats["source_url"] = info.get("source_url") or default_url

    def preserve_text(text: str, locator: str, role: str) -> None:
        text = clean_text(text)
        if text:
            record = make_record(path, document_id, relative_path, text, "txt",
                                 source_url=default_url, document_meta=info)
            record["record_id"] = f"{document_id}-{locator}"
            record["record_role"] = role
            records.append(record)

    if not headers:
        preserve_text(raw, "text_1", "body")
        if not records:
            warnings.append("Empty TXT document")
        return records, warnings, stats

    preserve_text(preamble, "text_preamble", "preamble")
    occurrences: Counter[int] = Counter()
    seen_content: set[tuple[int, str, str, str | None]] = set()
    numbers: set[int] = set()
    for position, header in enumerate(headers, start=1):
        number = int(header.group(1))
        numbers.add(number)
        stop = headers[position].start() if position < len(headers) else len(raw)
        block = raw[header.end():stop]
        qa = _QA_PATTERN.search(block)
        if qa is None:
            warnings.append(f"Incomplete FAQ {number}: preserved as unstructured text")
            preserve_text(raw[header.start():stop], f"text_faq_{position}", "incomplete_faq")
            continue
        question = clean_text(qa.group("question"))
        answer = clean_text(_TRAILING_SEPARATOR.sub("", qa.group("answer").rstrip()))
        local_url = _source_url(block, warnings, f"FAQ {number}") or default_url
        if not question or not answer or number < 1:
            warnings.append(f"Incomplete/invalid FAQ {number}: preserved as unstructured text")
            preserve_text(raw[header.start():stop], f"text_faq_{position}", "incomplete_faq")
            continue
        content_key = (number, question, answer, local_url)
        if content_key in seen_content:
            warnings.append(f"Exact repeated FAQ {number} skipped")
            continue
        seen_content.add(content_key)
        occurrences[number] += 1
        record = make_record(path, document_id, relative_path,
                             f"Question: {question}\nAnswer: {answer}", "faq",
                             faq_number=number, question=question,
                             source_url=local_url, document_meta=info)
        if occurrences[number] > 1:
            warnings.append(f"Duplicate FAQ number {number}: distinct content retained")
            record["record_id"] += f"-occurrence_{occurrences[number]}"
        records.append(record)
        # Do not silently drop material before a Question label in an FAQ block.
        preserve_text(block[:qa.start()], f"text_faq_prefix_{position}", "faq_prefix")
    gaps = _missing_faq_ranges(numbers)
    if gaps:
        warnings.append(f"Missing FAQ numbers: {gaps}")
    return records, warnings, stats


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Invalid JSON constant {value}")


def _load_overrides(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Load optional data/document_metadata.json, keyed by raw-relative paths."""
    metadata_path = Path(raw_dir).parent / "document_metadata.json"
    try:
        text = metadata_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    data = json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise ValueError("document_metadata.json must be an object keyed by relative paths")
    result: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        normalized = key.replace("\\", "/").strip()
        parts = PurePosixPath(normalized)
        if not normalized or parts.is_absolute() or ".." in parts.parts or ":" in normalized:
            raise ValueError(f"Metadata path must be relative to raw_dir: {key!r}")
        normalized = parts.as_posix()
        if normalized in result:
            raise ValueError(f"Duplicate normalized metadata path: {normalized}")
        result[normalized] = _validate_override(value, normalized)
    return result


def ingest(
    raw_dir: Path, processed_dir: Path, manifest_path: Path,
    *, allow_partial: bool = False,
) -> dict[str, Any]:
    """Extract PDFs/TXT recursively, including documents directly in raw_dir.

    By default, file errors, empty documents or failed PDF pages prevent
    publication. Inspect failed_ingestion_report.json and fix inputs, or
    explicitly use allow_partial=True to publish the usable subset.

    Run ingestion with indexing/query workers stopped: the four output paths
    are replaced individually, not as a multi-file transaction. The manifest
    is written last and includes a checksum for readers to verify the data.
    """
    raw_dir, processed_dir, manifest_path = map(Path, (raw_dir, processed_dir, manifest_path))
    if type(allow_partial) is not bool:
        raise TypeError("allow_partial must be a boolean")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw documents directory: {raw_dir}")
    overrides = _load_overrides(raw_dir)
    files = sorted(path for path in raw_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".pdf", ".txt"})
    if not files:
        raise RuntimeError(f"No PDF/TXT documents in {raw_dir}")
    outputs = [processed_dir / name for name in (
        "extracted.jsonl", "errors.json", "extraction_report.csv", "failed_ingestion_report.json"
    )] + [manifest_path]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Manifest path must differ from all generated report/data paths")
    if {path.resolve() for path in outputs} & {path.resolve() for path in files}:
        raise ValueError("Output paths must not overwrite input documents")

    extracted: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    report_rows: list[dict[str, Any]] = []
    seen_hashes: dict[str, tuple[str, dict[str, Any]]] = {}
    available_paths = {path.relative_to(raw_dir).as_posix() for path in files}
    warnings.extend({"file": key, "warning": "Metadata override does not match an input document"}
                    for key in sorted(overrides.keys() - available_paths))

    for path in files:
        relative_path = path.relative_to(raw_dir).as_posix()
        LOGGER.info("Processing: %s", relative_path)
        records: list[dict[str, Any]] = []
        file_warnings: list[str] = []
        stats: dict[str, Any] = {
            "total_pages": None, "empty_pages": None, "short_pages": None,
            "failed_pages": 0, "page_issues": [],
            **{key: None for key in _META_FIELDS}, "title_verified": False,
        }
        status, duplicate_of, file_hash = "ok", None, None
        try:
            file_hash = sha256_file(path)
            if file_hash in seen_hashes:
                duplicate_of, original_stats = seen_hashes[file_hash]
                stats = dict(original_stats)
                status = "duplicate"
                file_warnings.append(f"Exact duplicate of {duplicate_of}; indexed once")
                if relative_path in overrides:
                    file_warnings.append("Duplicate alias metadata is not applied to canonical records")
            else:
                extractor = extract_pdf if path.suffix.lower() == ".pdf" else extract_txt
                records, file_warnings, stats = extractor(path, file_hash, relative_path, overrides)
                # IDs must describe the bytes actually ingested, even if inputs change mid-run.
                if sha256_file(path) != file_hash:
                    raise RuntimeError("Source file changed during ingestion; retry with stable inputs")
                if records:
                    extracted.extend(records)
                    seen_hashes[file_hash] = (relative_path, dict(stats))
                    status = "warning" if file_warnings else "ok"
                else:
                    status = "error" if stats.get("failed_pages") else "empty"
                    file_warnings.append("No usable records extracted")
        except Exception as exc:
            records = []
            status = "error"
            file_warnings.append(f"{type(exc).__name__}: {exc}")
        inventory.append({
            "path": relative_path, "document_name": path.name, "sha256": file_hash,
            "source_type": path.suffix.lower().lstrip("."), "duplicate_of": duplicate_of,
            "status": status, "records": len(records),
            **{key: stats.get(key) for key in _META_FIELDS},
            "total_pages": stats.get("total_pages"),
            "page_issues": stats.get("page_issues", []),
        })
        warnings.extend({"file": relative_path, "warning": warning} for warning in file_warnings)
        report_rows.append({
            "document_name": path.name, "document_title": stats.get("document_title"),
            "title_source": stats.get("title_source"), "source_path": relative_path,
            "source_type": path.suffix.lower().lstrip("."),
            "total_pages": stats.get("total_pages"), "empty_pages": stats.get("empty_pages"),
            "short_pages": stats.get("short_pages"), "failed_pages": stats.get("failed_pages", 0),
            "records": len(records), "characters": sum(len(record["text"]) for record in records),
            "warnings": len(file_warnings), "status": status,
        })

    summary = {
        "source_count": len(inventory), "unique_source_count": len(seen_hashes),
        "record_count": len(extracted), "warning_count": len(warnings),
        "error_count": sum(row["status"] == "error" for row in report_rows),
        "empty_source_count": sum(row["status"] == "empty" for row in report_rows),
        "duplicate_count": sum(row["status"] == "duplicate" for row in report_rows),
        "failed_page_count": sum(row["failed_pages"] for row in report_rows if row["status"] != "duplicate"),
        "unverified_title_count": sum(
            source["status"] in {"ok", "warning"} and not source["title_verified"]
            for source in inventory
        ),
    }
    partial = bool(summary["error_count"] or summary["empty_source_count"] or summary["failed_page_count"])
    summary["partial"] = partial
    processed_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    if not extracted or (partial and not allow_partial):
        failure_path = processed_dir / "failed_ingestion_report.json"
        dump_json(failure_path, {"generated_at": generated_at, "summary": summary,
                                 "sources": inventory, "warnings": warnings})
        raise RuntimeError(
            f"Ingestion not published: {summary['record_count']} records, "
            f"{summary['error_count']} file errors, {summary['empty_source_count']} empty sources, "
            f"{summary['failed_page_count']} failed pages. Review {failure_path}. "
            "Previous extracted data and manifest were preserved."
        )

    record_ids = [record["record_id"] for record in extracted]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("Duplicate record IDs detected; ingestion not published")

    # Finish serialization before publishing any output. Each final replacement
    # is atomic; manifest checksum validation detects interrupted publication.
    with tempfile.TemporaryDirectory(prefix=".ingestion-", dir=processed_dir) as directory:
        staging = Path(directory)
        staged_records = staging / "extracted.jsonl"
        write_jsonl(staged_records, extracted)
        dump_json(staging / "errors.json", warnings)

        def write_report(handle: Any) -> None:
            writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
            writer.writeheader()
            writer.writerows(report_rows)

        _atomic_write(staging / "extraction_report.csv", write_report)
        manifest = {
            "version": "2.1", "generated_at": generated_at,
            "generated_by": "src/ingestion.py", "source_count": len(inventory),
            "unique_source_count": len(seen_hashes), "record_count": len(extracted),
            "extracted_sha256": sha256_file(staged_records), "partial": partial,
            "sources": inventory,
        }
        # Validate manifest serialization before replacing the extracted data.
        dump_json(staging / "manifest.json", manifest)
        for name in ("extracted.jsonl", "errors.json", "extraction_report.csv"):
            (staging / name).replace(processed_dir / name)
        dump_json(manifest_path, manifest)

    # A successful publication supersedes the diagnostic from an earlier run.
    # Failure to remove that diagnostic must not mark published data as failed.
    try:
        (processed_dir / "failed_ingestion_report.json").unlink(missing_ok=True)
    except OSError as exc:
        LOGGER.warning("Could not remove the previous failure diagnostic: %s", exc)

    LOGGER.info("Ingestion complete: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    from config import settings

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = ingest(settings.raw_dir, settings.processed_dir,
                    settings.root_dir / "data" / "manifest.json")
    print(json.dumps(result, indent=2))
