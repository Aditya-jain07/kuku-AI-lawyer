"""Brief safety notices, heuristic risk screening and citation diagnostics.

Detection does not establish intent or urgency. Citation identity checks do not
establish that a claim is supported, current or legally applicable.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any


# Category order is stable for existing callers. Patterns intentionally favour
# recall; quoted, educational, negated and historical statements can still match.
_URGENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "immediate_danger": (
        r"\bimmediate (?:physical )?danger\b",
        r"\bbeing attacked\b",
        r"\bthreaten(?:s|ed|ing)? (?:me|us|him|her|them) (?:with|to) (?:kill|hurt|attack)\b",
        r"\bthreaten(?:s|ed|ing)? to (?:kill|hurt|attack)(?: me| us| him| her| them)?\b",
        r"\bthreatened with violence\b",
        r"\bphysical assault\b",
        r"\bdomestic violence\b",
        r"\b(?:my|our) life is in danger\b",
        r"(?:मेरी|हमारी) जान (?:को खतरा|खतरे में)",
        r"(?:मुझे|हमें) (?:जान से )?मारने की धमकी",
        r"(?:मेरे|हमारे) साथ मारपीट",
    ),
    "custody_or_arrest": (
        r"\b(?:have been|has been|was|is|am|got|being) arrested\b",
        r"\bpolice custody\b",
        r"\bdetained (?:by|at) (?:the )?police\b",
        r"\b(?:my|our) (?:friend|relative|family member|brother|sister|father|mother|husband|wife) (?:has been|was|is) arrested\b",
        r"पुलिस (?:की )?हिरासत",
        r"गिरफ्तार (?:कर लिया|किया|हूँ|है|हो)",
    ),
    "self_harm": (
        r"\bsuicidal\b", r"\bsuicide\b", r"\bself[- ]harm\b",
        r"\bwant to (?:end my life|kill myself|die)\b",
        r"\bthinking (?:of|about) (?:hurting|killing) myself\b",
        r"आत्महत्या",
        r"जीना नहीं चाहत[ाी]",
        r"अपनी जान (?:लेना|देना) चाहत[ाी]",
    ),
    "cyber_fraud": (
        r"\bstolen otp\b",
        r"\b(?:bank |upi |payment )?account (?:was |is |got )?hacked\b",
        r"\bupi fraud\b", r"\b(?:online|banking|payment) scam\b", r"\bphishing\b",
        r"\bunauthori[sz]ed (?:bank |upi |card )?transaction\b",
        r"\b(?:money|funds) (?:was|were|has been|have been) (?:stolen|debited)\b",
        r"(?:यूपीआई|ऑनलाइन|बैंकिंग) (?:फ्रॉड|धोखाधड़ी|धोखाधड़ी)",
        r"(?:बैंक )?खाता हैक",
    ),
    "blackmail": (
        r"\bblackmail(?:ed|ing)?\b", r"\bextort(?:ion|ed|ing)?\b",
        r"\bthreaten(?:ed|ing)? to (?:leak|share|publish) (?:my|our) (?:photos|pictures|videos|images)\b",
        r"ब्लैकमेल",
        r"(?:फोटो|तस्वीरें|वीडियो).{0,40}(?:लीक|वायरल).{0,30}धमकी",
    ),
}
_COMPILED_PATTERNS = {
    category: tuple(re.compile(pattern, re.I) for pattern in patterns)
    for category, patterns in _URGENT_PATTERNS.items()
}

# ASCII IDs only; reject wrapped forms such as [[C1]] and leading zeros.
_CITATION_PATTERN = re.compile(r"(?<!\[)\[(C[1-9][0-9]*)\](?!\])")
_CITATION_ID = re.compile(r"C[1-9][0-9]*\Z")
_CITATION_LIKE = re.compile(r"\[+[ \t]*[Cc][ \t]*[0-9]+(?:[^\]\n]*\]+)?")

_DURATION_WORDS = {
    word: str(number) for number, word in enumerate(
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen", "twenty")
    )
}
_DURATION_WORDS.update({"thirty": "30", "forty": "40", "forty-five": "45",
                        "sixty": "60", "ninety": "90"})
_DURATION_BODY = (
    r"(?P<count>[0-9]+|" + "|".join(_DURATION_WORDS) + r")"
    r"\s+(?:(?P<kind>working|business|calendar)\s+)?"
    r"(?P<unit>days?|weeks?|months?|years?)\b"
)
_DEADLINE = re.compile(
    r"\b(?:within|no later than)\s+(?:a period of\s+)?" + _DURATION_BODY,
    re.I,
)
_DURATION = re.compile(r"\b" + _DURATION_BODY, re.I)


def citation_units(answer: str) -> list[str]:
    """Attach trailing markers to their sentence before splitting draft claims."""
    normalized = re.sub(
        r"([.!?।])\s*((?:\[C[1-9][0-9]*\](?:[ \t]+(?=\[C))?)+)",
        r" \2\1", answer,
    )
    return re.split(r"\n+|(?<=[.!?।])\s+", normalized)


def _durations(pattern: re.Pattern[str], text: str) -> set[tuple[str, str, str]]:
    values = set()
    for match in pattern.finditer(text):
        count = match["count"].casefold()
        number = _DURATION_WORDS.get(count, count.lstrip("0") or "0")
        kind = (match["kind"] or "").casefold()
        if kind == "business":
            kind = "working"
        values.add((number, match["unit"].casefold().rstrip("s"), kind))
    return values


def deadline_mentions(text: str) -> set[tuple[str, str, str]]:
    """Find common explicit English duration deadlines for citation screening."""
    return _durations(_DEADLINE, text)


def has_unmatched_cited_deadline(answer: str, passages: dict[str, str]) -> bool:
    """Screen cited deadlines for matching durations in their own cited text."""
    for unit in citation_units(answer):
        deadlines = deadline_mentions(unit)
        citations = extract_citations(unit)
        if not deadlines or not citations:
            continue
        supplied = set()
        for citation in citations:
            supplied.update(_durations(_DURATION, passages.get(citation, "")))
        if deadlines - supplied:
            return True
    return False


# Contact details verified against official Indian government sources.
# Emergency: https://112.gov.in/
# Financial cyber fraud: https://ncrp-grievanceredressal.mha.gov.in/
_NOTICE_TEXT: dict[str, str] = {
    "immediate_danger": (
        "If someone is in immediate physical danger, move to safety if possible "
        "and contact local emergency services; in India, call 112."
    ),
    "custody_or_arrest": (
        "If someone is currently detained or arrested, seek prompt help from "
        "a criminal lawyer or the relevant legal-aid authority."
    ),
    "self_harm": (
        "If there is an immediate risk of self-harm, contact emergency services "
        "or go to the nearest emergency department, and ask a trusted person to stay with you."
    ),
    "cyber_fraud": (
        "If money or account access may be compromised, contact your bank or payment provider "
        "immediately; in India, report financial cyber fraud at 1930 or https://cybercrime.gov.in."
    ),
    "blackmail": (
        "If you are being blackmailed, preserve messages and other evidence, "
        "avoid sharing more sensitive material, and seek police or legal assistance."
    ),
}


def urgent_flags(text: str) -> list[str]:
    """Return risk categories in stable order; this is screening, not triage."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = unicodedata.normalize("NFC", text)
    return [category for category, patterns in _COMPILED_PATTERNS.items()
            if any(pattern.search(text) for pattern in patterns)]


def safety_notice(text: str) -> str:
    """Use conditional, concise notices without claiming a risk is confirmed."""
    flags = urgent_flags(text)
    # A definition question alone is not evidence of a personal emergency.
    educational = bool(re.match(r"\s*(?:what is|what are|define|explain|meaning of)\b", text, re.I))
    personal = bool(re.search(r"\b(?:I|my|me|our|we|now|today)\b|मुझे|मेरी|मेरे|अभी", text, re.I))
    if educational and not personal:
        return ""
    priority = ("self_harm", "immediate_danger", "cyber_fraud", "blackmail", "custody_or_arrest")
    selected = next((flag for flag in priority if flag in flags), None)
    return _NOTICE_TEXT[selected] if selected else ""


def extract_citations(answer: str) -> list[str]:
    """Extract exact [C1] citations in order of first appearance."""
    if not isinstance(answer, str):
        return []
    return list(dict.fromkeys(_CITATION_PATTERN.findall(answer)))


def _allowed_ids(supplied_ids: Iterable[str]) -> set[str]:
    # Treat an accidentally supplied single ID as one ID, not its characters.
    if isinstance(supplied_ids, str):
        supplied_ids = (supplied_ids,)
    allowed: set[str] = set()
    for value in supplied_ids:
        if not isinstance(value, str) or not _CITATION_ID.fullmatch(value):
            raise ValueError("Supplied citation IDs must use the form C1, C2, etc.")
        allowed.add(value)
    return allowed


def validate_citations(answer: str, supplied_ids: Iterable[str]) -> list[str]:
    """Return cited IDs absent from supplied evidence; do not verify claims."""
    allowed = _allowed_ids(supplied_ids)
    # Numeric ordering without converting potentially enormous strings to int.
    return sorted((cid for cid in extract_citations(answer) if cid not in allowed),
                  key=lambda cid: (len(cid), cid))


def citation_audit(answer: str, supplied_ids: Iterable[str]) -> dict[str, Any]:
    """Audit identities, missing citations and malformed citation-like groups.

    citation_check_passed is a formatting/identity result, never an entailment
    or legal-accuracy check. all_cited_ids_supplied preserves its original
    meaning, including True when no IDs were cited.
    """
    allowed = _allowed_ids(supplied_ids)
    cited = extract_citations(answer)
    invalid = validate_citations(answer, allowed)
    valid = [cid for cid in cited if cid in allowed]
    malformed = list(dict.fromkeys(
        match.group() for match in _CITATION_LIKE.finditer(answer if isinstance(answer, str) else "")
        if not _CITATION_PATTERN.fullmatch(match.group())
    ))
    return {
        "cited_ids": cited, "invalid_ids": invalid,
        "has_citations": bool(cited), "all_cited_ids_supplied": not invalid,
        "valid_ids": valid, "malformed_citations": malformed,
        "has_supplied_evidence": bool(allowed),
        "missing_citations": bool(allowed) and not valid,
        "citation_check_passed": bool(valid) and not invalid and not malformed,
    }


def _positive_locator(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,9}", value.strip()):
        number = int(value)
        return number if number > 0 else None
    return None


def _page_label(pages: list[int]) -> str:
    ordered = sorted(set(pages))
    spans: list[str] = []
    start = end = ordered[0]
    for page in ordered[1:]:
        if page == end + 1:
            end = page
        else:
            spans.append(str(start) if start == end else f"{start}–{end}")
            start = end = page
    spans.append(str(start) if start == end else f"{start}–{end}")
    return ("page " if len(ordered) == 1 else "pages ") + ", ".join(spans)


def _label_text(value: Any) -> str:
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def source_label(metadata: dict[str, Any]) -> str:
    """Keep titles, original filenames and exact physical-page provenance."""
    if not isinstance(metadata, dict):
        return "Unknown source"
    filename = _label_text(metadata.get("document_name")) or "Unknown source"
    title = _label_text(metadata.get("document_title"))
    name = f"{title} ({filename})" if title and title.casefold() != filename.casefold() else filename
    parts = [name]
    section = _label_text(metadata.get("section"))
    if section and not section.casefold().startswith("faq"):
        parts.append(section)

    explicit_pages = metadata.get("page_numbers")
    pages = [_positive_locator(page) for page in explicit_pages] if isinstance(explicit_pages, (list, tuple)) else []
    valid_pages = [page for page in pages if page is not None]
    if valid_pages:
        parts.append(_page_label(valid_pages))
    else:
        start = _positive_locator(metadata.get("page_start"))
        end = _positive_locator(metadata.get("page_end"))
        number = _positive_locator(metadata.get("page_number"))
        if start is not None:
            parts.append(f"pages {start}–{end}" if end is not None and end > start else f"page {start}")
        elif number is not None:
            parts.append(f"page {number}")
    faq = _positive_locator(metadata.get("faq_number"))
    if faq is not None:
        parts.append(f"FAQ {faq}")
    elif section.casefold().startswith("faq"):
        parts.append(section)
    return ", ".join(parts)


def evidence_warning(retrieved_chunks: list[Any]) -> str:
    """Report absent evidence without implying that retrieved evidence suffices."""
    if not retrieved_chunks:
        return (
            "I could not find supporting material in the available documents. "
            "Please clarify the issue or check an official source or qualified legal professional."
        )
    return ""
