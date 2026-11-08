"""Source-aware, token-checked legal passages for Kuku's E5/FAISS index.

Keep this module in src/ alongside utils.py. Chunk text excludes E5's
``passage: `` prefix; token_count includes that prefix and special tokens.
The embedding module must add the prefix once and verify its own input limit.
"""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import lru_cache
from typing import Any

from config import settings
from .utils import clean_text


MAX_EMBEDDING_TOKENS = 512  # Default for the selected E5-small model.
TOKEN_SAFETY_MARGIN = 8
PASSAGE_PREFIX = "passage: "
_TOKENIZE_CHARS = 400
_SECTION_LINE = re.compile(
    r"^[ \t]*(?:(?P<kind>Section|Article|Rule|Regulation)[ \t]+)?"
    r"(?P<number>[0-9]{1,4}[A-Z]{0,2})"
    r"(?:[ \t]*(?P<punctuation>\.(?![0-9])|[\-–—:])[ \t]*|[ \t]+)"
    r"(?P<heading>[^\n]+)$", re.I,
)
_HEADING_BODY_SEPARATOR = re.compile(r"\.\s*[—–-]\s*|[—–]\s*(?=[(\[0-9A-Z])")
_TOC_START = re.compile(
    r"^[ \t]*(?:ARRANGEMENT OF (?:SECTIONS|ARTICLES|RULES|REGULATIONS)|"
    r"TABLE OF CONTENTS|CONTENTS)[ \t]*[.:-]?[ \t]*$", re.I,
)
_TOC_END = re.compile(
    r"\b(?:BE IT ENACTED|AN ACT TO|WE, THE PEOPLE|PREAMBLE)\b|"
    r"^[ \t]*(?:THE )?.{3,100}\b(?:ACT|CODE|SANHITA|RULES|REGULATIONS)\b.{0,20}"
    r"\b(?:18|19|20)[0-9]{2}[ \t]*$", re.I,
)
_BODY_LANGUAGE = re.compile(r"\b(?:shall|hereby|provided that|this act|this code)\b", re.I)
_FAQ_BODY = re.compile(r"\AQuestion:[ \t]*(.*?)\nAnswer:[ \t]*(.*)\Z", re.I | re.S)


@lru_cache(maxsize=2)
def _load_tokenizer(model_name: str):
    # Lazy import: importing utilities/chunking does not download a model.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=False,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast embedding tokenizer is required for source offsets")
    return tokenizer


def get_tokenizer():
    """Cache by model ID, so a changed model cannot reuse a stale tokenizer."""
    return _load_tokenizer(settings.embedding_model)


def _token_count(text: str, *, add_special_tokens: bool = True) -> int:
    return len(get_tokenizer().encode(
        text, add_special_tokens=add_special_tokens, truncation=False, verbose=False,
    ))


def _validate_budgets(target_tokens: int, overlap_tokens: int) -> None:
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError("target_tokens must be a positive integer")
    if type(overlap_tokens) is not int or not 0 <= overlap_tokens < target_tokens:
        raise ValueError("overlap_tokens must be an integer from 0 to target_tokens - 1")


def _embedding_budget(target_tokens: int) -> int:
    limit = getattr(settings, "embedding_max_tokens", MAX_EMBEDDING_TOKENS)
    declared = getattr(get_tokenizer(), "model_max_length", None)
    # Hugging Face uses extremely large sentinel values for unknown limits.
    if isinstance(declared, int) and not isinstance(declared, bool) and 0 < declared < 1_000_000:
        limit = min(limit, declared)
    budget = min(target_tokens, limit - TOKEN_SAFETY_MARGIN)
    if budget <= _token_count(PASSAGE_PREFIX):
        raise ValueError("Embedding budget has no room for source text after prefix/special tokens")
    return budget


def _token_offsets(text: str) -> list[tuple[int, int]]:
    """Build character anchors using bounded tokenizer calls.

    Anchors estimate a window, not its exact token count: fragment boundaries
    can change tokenization, and multiple tokens can map to one Unicode span.
    Every final passage is counted again using the complete embedding input.
    """
    tokenizer = get_tokenizer()
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for source offsets")
    offsets: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        limit = min(cursor + _TOKENIZE_CHARS, len(text))
        if limit < len(text):
            boundary = max(text.rfind(" ", cursor + 200, limit),
                           text.rfind("\n", cursor + 200, limit))
            if boundary > cursor:
                limit = boundary + 1
        fragment = text[cursor:limit]
        data = tokenizer(fragment, add_special_tokens=False,
                         return_offsets_mapping=True, truncation=False, verbose=False)
        for start, end in data["offset_mapping"]:
            if end <= start:
                continue
            if not 0 <= start < end <= len(fragment):
                raise ValueError("Tokenizer returned an invalid source offset")
            span = (cursor + start, cursor + end)
            if offsets and span[0] < offsets[-1][1]:
                # Merge overlapping Unicode spans rather than splitting a
                # character between chunks or relying on duplicate anchors.
                offsets[-1] = (min(offsets[-1][0], span[0]), max(offsets[-1][1], span[1]))
            else:
                offsets.append(span)
        cursor = limit
    if text.strip() and not offsets:
        raise ValueError("Nonempty source text produced no tokenizer offsets")
    return offsets


def _pdf_page(record: dict[str, Any]) -> int:
    page = record.get("page_number")
    if type(page) is not int or page < 1:
        raise ValueError("PDF records require positive, one-based integer page_number values")
    return page


def _prepare_pdf_document(
    records: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int, int]]]:
    """Join a contiguous page run and retain cleaned-text character ranges."""
    parts: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    position = 0
    previous: int | None = None
    for record in sorted(records, key=_pdf_page):
        page = _pdf_page(record)
        body = clean_text(record.get("text", ""))
        if not body:
            continue
        if previous is not None and page != previous + 1:
            raise ValueError("PDF pages must be unique and consecutive within a chunking run")
        if parts:
            parts.append("\n\n")
            position += 2
        start = position
        parts.append(body)
        position += len(body)
        ranges.append((start, position, page))
        previous = page
    return "".join(parts), ranges


class _PageIndex:
    def __init__(self, ranges: list[tuple[int, int, int]]):
        self.ranges = ranges
        self.starts = [start for start, _, _ in ranges]
        self.ends = [end for _, end, _ in ranges]
        previous_end = 0
        for start, end, page in ranges:
            if start < previous_end or end <= start or type(page) is not int or page < 1:
                raise ValueError("Invalid or overlapping PDF page ranges")
            previous_end = end

    def pages(self, start: int, end: int) -> list[int]:
        if end <= start:
            return []
        left = bisect_right(self.ends, start)
        right = bisect_left(self.starts, end)
        return [page for _, _, page in self.ranges[left:right]]


def _locate_pages(
    start: int, end: int, page_ranges: list[tuple[int, int, int]],
) -> tuple[int | None, int | None]:
    pages = _PageIndex(page_ranges).pages(start, end)
    return (min(pages), max(pages)) if pages else (None, None)


def _section_markers(text: str, *, default_kind: str = "Section") -> list[tuple[int, str]]:
    """Find heading candidates; avoid obvious TOCs and subsection/list lines."""
    markers: list[tuple[int, str]] = []
    position, in_toc = 0, False
    for line in text.splitlines(keepends=True):
        if _TOC_START.match(line.strip()):
            in_toc = True
        match = _SECTION_LINE.match(line.rstrip("\n"))
        if match and not (match.group("kind") or match.group("punctuation")):
            match = None
        separator = _HEADING_BODY_SEPARATOR.search(match.group("heading")) if match else None
        # A TOC entry may end in ".—". Require text after that separator
        # before treating it as the start of a provision.
        if in_toc and (
            _TOC_END.search(line)
            or (separator and match.group("heading")[separator.end():].strip())
        ):
            in_toc = False
        if match and not in_toc:
            raw = _HEADING_BODY_SEPARATOR.split(match.group("heading"), maxsplit=1)[0]
            raw = raw.strip(" .\t-–—")
            letters = [char for char in raw if char.isalpha()]
            first = letters[0] if letters else ""
            if (3 <= len(raw) <= 110 and len(raw.split()) <= 16 and len(letters) >= 3
                    and not raw.startswith(("(", "[", "{", ":"))
                    and not re.search(r"\.{2,}|[;?]|\.\s|\s[0-9]+$", raw)
                    and not _BODY_LANGUAGE.search(raw)
                    and (match.group("kind") or not first.islower())):
                kind = match.group("kind") or default_kind
                label = f"{kind.title()} {match.group('number').upper()} — {raw}"
                markers.append((position + match.start(), label))
        position += len(line)
    return markers


def _one_line(value: Any, limit: int | None) -> str:
    value = re.sub(r"\s+", " ", clean_text(value if isinstance(value, str) else "")).strip()
    if limit is not None and len(value) > limit:
        return value[:limit].rstrip() + "…"
    return value


def _header(
    metadata: dict[str, Any], section: str | None, question: str | None,
    *, field_limit: int | None = None,
) -> str:
    """Create compact retrieval context; keep full metadata separately."""
    lines: list[str] = []
    title = _one_line(metadata.get("document_title"), field_limit)
    filename = _one_line(metadata.get("document_name"), field_limit)
    if title and title.lower() not in {filename.lower(), filename.rsplit(".", 1)[0].lower()}:
        lines.append(f"Document: {title}")
    elif filename:
        lines.append(f"Source file: {filename}")
    section = _one_line(section, field_limit)
    if section:
        lines.append(section if re.match(r"(?:Section|Article|Rule|Regulation|FAQ)\b", section, re.I)
                     else f"Section: {section}")
    question = _one_line(question, field_limit)
    if question:
        lines.extend((f"Question: {question}", "Answer:"))
    return "\n".join(lines) + "\n" if lines else ""


def _fit_header(
    metadata: dict[str, Any], section: str | None, question: str | None, budget: int,
) -> tuple[str, bool]:
    full = _header(metadata, section, question)
    header_budget = max(_token_count(PASSAGE_PREFIX), min(96, budget // 3))
    # Only bounded strings are tokenized, even if a supplied title is enormous.
    for limit in (240, 160, 96, 64, 32, 16, 8):
        prefix = _header(metadata, section, question, field_limit=limit)
        if _token_count(PASSAGE_PREFIX + prefix) <= header_budget:
            return prefix, prefix != full
    return "", bool(full)


def _preferred_end(text: str, start: int, end: int) -> int:
    """Prefer a nearby paragraph/sentence/line boundary over a clipped clause."""
    lower = start + max(1, int((end - start) * 0.60))
    boundary = text.rfind("\n\n", lower, end)
    if boundary >= lower:
        return boundary
    sentences = list(re.finditer(r"[.!?।][\"'’”)\]]*(?=\s|$)", text[lower:end]))
    if sentences:
        return lower + sentences[-1].end()
    boundary = text.rfind("\n", lower, end)
    if boundary >= lower:
        return boundary
    if end < len(text) and text[end - 1:end].isalnum() and text[end:end + 1].isalnum():
        boundary = text.rfind(" ", lower, end)
        if boundary >= lower:
            return boundary
    return end


def _fit_window(text: str, start: int, end: int, prefix: str, budget: int) -> tuple[int, int]:
    overhead = _token_count(PASSAGE_PREFIX + prefix)
    while True:
        count = _token_count(PASSAGE_PREFIX + prefix + text[start:end].strip())
        if count <= budget:
            break
        if end - start <= 1:
            raise ValueError("A source character cannot fit into the configured embedding budget")
        fraction = max(0.05, (budget - overhead) / max(1, count - overhead) * 0.90)
        end = start + max(1, min(end - start - 1, int((end - start) * fraction)))
    if end == len(text):
        return end, count
    preferred = _preferred_end(text, start, end)
    if start < preferred < end:
        preferred_count = _token_count(PASSAGE_PREFIX + prefix + text[start:preferred].strip())
        if preferred_count <= budget:
            return preferred, preferred_count
    return end, count


def _overlap_start(
    text: str, start: int, end: int, offsets: list[tuple[int, int]],
    starts: list[int], requested: int,
) -> int:
    if requested == 0:
        return end
    left, right = bisect_left(starts, start), bisect_left(starts, end)
    # Cap effective overlap at one third of the available anchor spans. This
    # guarantees useful forward progress when a small window has a large header.
    amount = min(requested, max(0, (right - left) // 3))
    index = max(left + 1, right - amount)
    while index < right:
        candidate = offsets[index][0]
        if start < candidate < end and _token_count(text[candidate:end], add_special_tokens=False) <= requested:
            return candidate
        index += 1
    return end


def _chunk_text(
    text: str, metadata: dict[str, Any], target_tokens: int, overlap_tokens: int,
    page_ranges: list[tuple[int, int, int]] | None = None,
    question: str | None = None,
) -> list[dict[str, Any]]:
    """Create passages without dropping source text or crossing section cuts."""
    _validate_budgets(target_tokens, overlap_tokens)
    cleaned = clean_text(text)
    if page_ranges and cleaned != text:
        raise ValueError("PDF character ranges must refer to already-cleaned source text")
    text = cleaned
    if not text:
        return []
    document_id = metadata.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Every chunk requires a nonempty document_id")
    budget = _embedding_budget(target_tokens)
    is_pdf = metadata.get("source_type") == "pdf"
    default_kind = "Article" if "constitution" in str(metadata.get("document_title", "")).lower() else "Section"
    # Numbered FAQ/list entries in guides must not become legal sections.
    source_path = str(metadata.get("relative_source_path", ""))
    is_legal = (metadata.get("document_type") == "legal_act" or
                source_path.partition("/")[0].casefold() == "acts")
    markers = _section_markers(text, default_kind=default_kind) if is_pdf and is_legal else []
    detected_sections = {label for _, label in markers}
    cuts: list[tuple[int, str | None]] = [(0, metadata.get("section"))]
    for position, label in markers:
        if position == cuts[-1][0]:
            cuts[-1] = (position, label)
        else:
            cuts.append((position, label))
    page_index = _PageIndex(page_ranges or [])
    output: list[dict[str, Any]] = []
    page_record_ids = metadata.get("_page_record_ids", {})
    public_metadata = {key: value for key, value in metadata.items() if not key.startswith("_")}

    for cut_index, (section_start, section) in enumerate(cuts):
        section_end = cuts[cut_index + 1][0] if cut_index + 1 < len(cuts) else len(text)
        segment = text[section_start:section_end]
        if not segment.strip():
            continue
        offsets = _token_offsets(segment)
        starts = [start for start, _ in offsets]
        ends = [end for _, end in offsets]
        prefix, header_truncated = _fit_header(metadata, section, question, budget)
        room = max(1, budget - _token_count(PASSAGE_PREFIX + prefix) - 2)
        content_end = len(segment.rstrip())
        cursor = 0
        while cursor < content_end:
            first = min(bisect_right(ends, cursor), len(offsets) - 1)
            stop = min(first + room, len(offsets))
            candidate_end = len(segment) if stop == len(offsets) else offsets[stop][0]
            # Bound full-passage counting too, including pathological long words.
            candidate_end = min(candidate_end, cursor + max(400, budget * 8))
            candidate_end = max(cursor + 1, candidate_end)
            end, count = _fit_window(segment, cursor, candidate_end, prefix, budget)
            raw_body = segment[cursor:end]
            body = raw_body.strip()
            if body:
                char_start = section_start + cursor + len(raw_body) - len(raw_body.lstrip())
                char_end = section_start + end - len(raw_body) + len(raw_body.rstrip())
                pages = page_index.pages(char_start, char_end)
                if not pages and metadata.get("page_number") is not None:
                    pages = [metadata["page_number"]]
                record_ids = [page_record_ids[page] for page in pages if page in page_record_ids]
                if not record_ids and metadata.get("record_id"):
                    record_ids = [metadata["record_id"]]
                content = prefix + body
                # Record identity separates repeated FAQ numbers and TXT pieces.
                identity = json.dumps([document_id, record_ids, char_start, char_end, content],
                                      ensure_ascii=False, separators=(",", ":"))
                digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                locator = (f"p{pages[0]}-{pages[-1]}" if pages else
                           f"faq{metadata['faq_number']}" if metadata.get("faq_number") is not None else "text")
                chunk = {
                    **public_metadata,
                    "chunk_id": f"{document_id}-{locator}-{digest}",
                    "chunk_number": len(output) + 1,
                    "page_number": pages[0] if pages else None,
                    "page_start": pages[0] if pages else None,
                    "page_end": pages[-1] if pages else None,
                    "page_numbers": pages, "source_record_ids": record_ids,
                    "section": section,
                    "section_source": "detected_candidate" if section in detected_sections else metadata.get("section_source"),
                    "text": content, "body_text": body,
                    "char_count": len(content), "token_count": count,
                    # Offsets refer to the cleaned record or contiguous PDF run.
                    "body_char_start": char_start, "body_char_end": char_end,
                    "context_header_truncated": header_truncated,
                }
                if record_ids:
                    chunk["record_id"] = record_ids[0]
                output.append(chunk)
            if end >= content_end:
                break
            next_cursor = _overlap_start(segment, cursor, end, offsets, starts, overlap_tokens)
            if not cursor < next_cursor <= end:
                raise RuntimeError("Chunk overlap failed to make forward progress")
            cursor = next_cursor
    return output


def chunk_record(
    record: dict[str, Any], target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Chunk one PDF page, FAQ or text record; build_chunks joins PDF pages."""
    target = settings.chunk_target_tokens if target_tokens is None else target_tokens
    overlap = settings.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens
    _validate_budgets(target, overlap)
    text = clean_text(record.get("text", ""))
    if not text:
        return []
    metadata = {key: value for key, value in record.items() if key != "text"}
    question = None
    if record.get("source_type") == "faq":
        match = _FAQ_BODY.match(text)
        if match:
            question, text = clean_text(match.group(1)), clean_text(match.group(2))
        else:
            question = clean_text(record.get("question") or "") or None
        if question:
            metadata["question"] = question
    ranges = None
    if record.get("source_type") == "pdf":
        page = _pdf_page(record)
        ranges = [(0, len(text), page)]
        if record.get("record_id"):
            metadata["_page_record_ids"] = {page: record["record_id"]}
    return _chunk_text(text, metadata, target, overlap, ranges, question)


def build_chunks(
    records: list[dict[str, Any]], target_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Join only consecutive PDF pages; keep FAQ/TXT records independent."""
    target = settings.chunk_target_tokens if target_tokens is None else target_tokens
    overlap = settings.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens
    _validate_budgets(target, overlap)
    pdf_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    others: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Ingestion records must be dictionaries")
        if not isinstance(record.get("document_id"), str) or not record["document_id"].strip():
            raise ValueError("Ingestion records require nonempty document_id values")
        if record.get("source_type") == "pdf":
            _pdf_page(record)
            pdf_groups[record["document_id"]].append(record)
        else:
            others.append(record)

    chunks: list[dict[str, Any]] = []
    for pages in pdf_groups.values():
        ordered = sorted(pages, key=_pdf_page)
        numbers = [_pdf_page(page) for page in ordered]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"Duplicate PDF page numbers for {ordered[0]['document_id']}")
        for field in ("document_title", "document_name", "document_type", "title_verified",
                      "title_source", "relative_source_path", "source_url", "jurisdiction", "effective_date"):
            if any(page.get(field) != ordered[0].get(field) for page in ordered[1:]):
                raise ValueError(f"Inconsistent PDF metadata field {field!r}")
        runs: list[list[dict[str, Any]]] = []
        for page in ordered:
            if not clean_text(page.get("text", "")):
                continue
            if not runs or _pdf_page(page) != _pdf_page(runs[-1][-1]) + 1:
                runs.append([])
            runs[-1].append(page)
        for run in runs:
            document, ranges = _prepare_pdf_document(run)
            metadata = {key: value for key, value in run[0].items() if key != "text"}
            metadata["section"] = None
            metadata["_page_record_ids"] = {
                _pdf_page(page): page["record_id"] for page in run if page.get("record_id")
            }
            chunks.extend(_chunk_text(document, metadata, target, overlap, ranges))
    for record in others:
        chunks.extend(chunk_record(record, target, overlap))

    seen: set[str] = set()
    numbers: dict[str, int] = defaultdict(int)
    budget = _embedding_budget(target) if chunks else target
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in seen:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")
        seen.add(chunk_id)
        numbers[chunk["document_id"]] += 1
        chunk["chunk_number"] = numbers[chunk["document_id"]]
        if chunk["token_count"] > budget:
            raise ValueError(f"Oversized chunk: {chunk_id}")
    return chunks
