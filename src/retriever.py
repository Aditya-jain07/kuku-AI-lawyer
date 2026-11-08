"""Hybrid legal retrieval for Kuku: E5/FAISS + BM25 + RRF + reranking.

Compatibility: Retriever(dir, model, top_k_candidates, top_k_context),
Retriever.ready, Retriever.metadata, Retriever.search(), RetrievedChunk.
No embedding/index rebuild is needed solely for BM25; it indexes metadata in RAM.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import settings
from .utils import read_jsonl, sha256_file


@dataclass
class RetrievedChunk:
    citation_id: str
    score: float
    text: str
    metadata: dict[str, Any]


JURISDICTIONS = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar",
    "chhattisgarh", "delhi", "goa", "gujarat", "haryana",
    "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "chandigarh", "ladakh",
    "jammu and kashmir", "puducherry", "lakshadweep",
    "andaman and nicobar islands",
    "dadra and nagar haveli and daman and diu",
]

# Do not create an inferred legal jurisdiction from user location or keywords.
# Only metadata that explicitly states a *single* applicable jurisdiction is
# used for exclusions. Empty/uncertain metadata is retained, not validated.

_STATE_PATTERNS = [(state, re.compile(rf"(?<!\w){re.escape(state)}(?!\w)", re.I))
                   for state in sorted(JURISDICTIONS, key=len, reverse=True)]
_SINGLE_SCOPES = {
    label: state for state in JURISDICTIONS
    for label in (state, f"state: {state}", f"state of {state}", f"union territory: {state}")
}


@lru_cache(maxsize=512)
def _state_from_text(text: str) -> str | None:
    """A multi-state question must not be filtered to whichever name comes first."""
    matches = [state for state, pattern in _STATE_PATTERNS if pattern.search(text)]
    return matches[0] if len(matches) == 1 else None


def _compatible(chunk: dict[str, Any], query: str) -> bool:
    # A place mentioned in free text may describe the employer, a comparison,
    # or a previous incident. Leave applicability to explicit metadata/context;
    # never silently discard legal evidence from this inference alone.
    return True

_STOPWORDS = frozenset("""
    a an the and or to of for in on at by from with as is are was were
    be been being do does did can could should would will have has had
    i me my we our you your he she they them their this that those these
    it its what which where when how why who whom about into out up down
    please tell explain help need want
""".split())


def _tokens(value: str) -> list[str]:
    """Keep combining marks attached to Hindi words and retain section digits."""
    text = unicodedata.normalize("NFC", str(value)).casefold()
    terms: list[str] = []
    word: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"L", "N"} or (word and (category[0] == "M" or char in "\u200c\u200d")):
            word.append(char)
        elif word:
            terms.append("".join(word))
            word = []
    if word:
        terms.append("".join(word))
    return [term for term in terms if term not in _STOPWORDS]




# Domain-specific expansions aid lexical recall. They do NOT assert that
# any particular statute applies. Keep terms short and conservative.
_EXPANSIONS: tuple[tuple[set[str], tuple[str, ...]], ...] = (
    ({"salary", "salaries", "wage", "wages", "paycheck", "payroll", "remuneration", "वेतन", "तनख्वाह", "सैलरी"},
     ("wages", "salary", "remuneration", "payment", "employee", "employer")),
    ({"unpaid", "nonpayment", "arrears", "delayed", "withheld"},
     ("non payment", "delay", "deduction", "arrears", "payment")),
    ({"landlord", "tenant", "rent", "rental", "lease", "eviction", "किराया", "किरायेदार", "मकानमालिक"},
     ("tenancy", "rent", "landlord", "tenant", "eviction", "deposit")),
    ({"refund", "seller", "defective", "consumer", "product", "रिफंड", "उपभोक्ता", "दोषपूर्ण"},
     ("consumer", "complaint", "refund", "deficiency", "service")),
    ({"upi", "transaction", "bank", "banking", "debit", "fraud"},
     ("bank", "transaction", "payment", "complaint", "unauthorised")),
    ({"cybercrime", "phishing", "hacked", "cyber"},
     ("cyber", "fraud", "complaint", "information", "technology")),
    ({"aid", "nalsa", "lawyer", "advocate"},
     ("legal", "aid", "services", "authority", "assistance")),
)


def _query_terms(query: str, expand: bool) -> list[str]:
    original = _tokens(query)
    if not expand:
        return list(dict.fromkeys(original))
    found = set(original)
    # A standalone deposit may be a bank deposit. A rental refund may
    # describe a tenant dispute rather than a consumer purchase.
    if {"security", "deposit"} <= found:
        found.add("rental")
    rental = bool(found & {"rent", "rental", "lease", "landlord", "tenant",
                           "किराया", "किरायेदार", "मकानमालिक"})
    if rental:
        found.discard("refund")
    terms = list(original)
    for triggers, synonyms in _EXPANSIONS:
        if found & triggers:
            terms.extend(term for synonym in synonyms for term in _tokens(synonym))
    return list(dict.fromkeys(terms))


def _searchable_text(chunk: dict[str, Any]) -> str:
    # New chunks expose body_text; include full metadata once, not the same
    # title/question twice. Keep support for legacy chunks containing text only.
    title = chunk.get("document_title") or chunk.get("document_name") or ""
    heading = chunk.get("section_heading") or chunk.get("section") or ""
    question = chunk.get("question") or ""
    body = chunk.get("body_text") or chunk.get("text") or ""
    return f"{title}\n{heading}\n{question}\n{body}"

@lru_cache(maxsize=2)
def _reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name, device="cpu", max_length=512)


class Retriever:
    def __init__(
        self,
        vector_store_dir: Path,
        model_name: str,
        top_k_candidates: int | None = None,
        top_k_context: int | None = None,
    ):
        self.vector_store_dir = Path(vector_store_dir)
        self.model_name = model_name
        top_k_candidates = settings.top_k_candidates if top_k_candidates is None else top_k_candidates
        top_k_context = settings.top_k_context if top_k_context is None else top_k_context
        if (type(top_k_candidates) is not int or type(top_k_context) is not int
                or top_k_candidates < 1 or top_k_context < 1):
            raise ValueError("Retrieval limits must be positive integers")

        # Explicit constructor parameters win, retaining compatibility
        # with AssistantService and existing smoke tests.
        self.top_k_candidates = top_k_candidates
        self.top_k_context = top_k_context
        self.bm25_top_k = settings.bm25_top_k
        self.rerank_top_n = max(top_k_context, settings.rerank_top_n)
        self.rrf_k = settings.rrf_k
        self.enable_query_expansion = settings.enable_query_expansion
        self.enable_reranking = settings.enable_reranking
        self.reranker_model = settings.reranker_model
        self.reranker_batch_size = settings.reranker_batch_size

        self.index = None
        self.manifest: dict[str, Any] = {}
        self._lexical_only = False
        self._semantic_unavailable = False
        self._reranking_succeeded = False
        self.metadata: list[dict[str, Any]] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_term_sets: list[set[str]] = []
        self._reranking_unavailable = False
        self._load()
        if self.metadata:
            self._build_bm25()

    @property
    def ready(self) -> bool:
        """True when semantic retrieval or the explicit BM25-only fallback is usable."""
        return bool(self.metadata) and (
            (self.index is not None and self.index.ntotal == len(self.metadata))
            or (self.index is None and self._lexical_only)
        )

    def _load(self) -> None:
        index_path = self.vector_store_dir / "index.faiss"
        metadata_path = self.vector_store_dir / "metadata.jsonl"
        manifest_path = self.vector_store_dir / "index_manifest.json"
        if (self.vector_store_dir / ".index-build.lock").exists():
            raise RuntimeError("Index is being rebuilt; initialize Retriever after the build finishes")
        if not index_path.exists() and not metadata_path.exists():
            if manifest_path.exists():
                raise RuntimeError("Index manifest exists but index data and metadata are missing")
            return
        if not metadata_path.is_file():
            raise RuntimeError("FAISS index exists but metadata.jsonl is missing")

        if index_path.is_file() and manifest_path.is_file():
            from .embeddings import load_faiss_index
            self.index, self.metadata, self.manifest = load_faiss_index(
                self.vector_store_dir, self.model_name,
            )
            return

        if manifest_path.exists():
            raise RuntimeError("Index generation is incomplete; rebuild before retrieval")

        # Preserve legacy stores and the original BM25-only fallback. New
        # generations always go through the verified loader above.
        self.metadata = read_jsonl(metadata_path, missing_ok=False)
        if not self.metadata:
            raise RuntimeError("metadata.jsonl contains no records")
        ids: set[str] = set()
        for row in self.metadata:
            chunk_id = row.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in ids:
                raise RuntimeError("Missing or duplicate chunk_id in index metadata")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise RuntimeError(f"Empty chunk text: {chunk_id}")
            ids.add(chunk_id)
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if not isinstance(self.manifest, dict):
                raise RuntimeError("Index manifest must be a JSON object")
            if self.manifest.get("chunk_count") != len(self.metadata):
                raise RuntimeError("Metadata count differs from index manifest")
            expected = self.manifest.get("metadata_sha256")
            if expected and _sha256(metadata_path) != expected:
                raise RuntimeError("Metadata checksum mismatch; rebuild index")
        if not index_path.exists():
            self._lexical_only = True
            warnings.warn("FAISS missing; using BM25-only retrieval", stacklevel=2)
            return
        import faiss
        warnings.warn("Legacy index has no manifest; rebuild it to enable checksum verification", stacklevel=2)
        self.index = faiss.read_index(str(index_path))
        if (self.index.ntotal != len(self.metadata)
                or self.index.metric_type != faiss.METRIC_INNER_PRODUCT):
            raise RuntimeError("Legacy FAISS index count/metric does not match expected metadata")



    def _build_bm25(self) -> None:
        """Build an in-memory inverted index; no additional dependency."""
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_lengths = []
        self._doc_term_sets = []
        for row_id, chunk in enumerate(self.metadata):
            counts = Counter(_tokens(_searchable_text(chunk)))
            self._doc_lengths.append(sum(counts.values()))
            self._doc_term_sets.append(set(counts))
            for term, count in counts.items():
                postings[term].append((row_id, count))
        self._postings = dict(postings)
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths else 1.0
        )

    def _bm25_search(self, query: str, candidate_count: int) -> list[int]:
        terms = _query_terms(query, self.enable_query_expansion)
        if not terms or not self.metadata:
            return []
        document_count = len(self.metadata)
        k1, b = 1.5, 0.75
        scores: dict[int, float] = defaultdict(float)
        state_query = query
        original_terms = set(_tokens(query))
        for term in terms:
            weight = 1.0 if term in original_terms else 0.35
            postings = self._postings.get(term, ())
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            for row_id, tf in postings:
                if not _compatible(self.metadata[row_id], state_query):
                    continue
                length = self._doc_lengths[row_id]
                norm = k1 * (1 - b + b * length / max(self._avg_doc_length, 1))
                scores[row_id] += weight * idf * (tf * (k1 + 1)) / (tf + norm)
        return [idx for idx, _ in sorted(
            scores.items(), key=lambda row: (-row[1], row[0])
        )[:candidate_count]]

    def _semantic_search(self, query: str, candidate_count: int) -> list[int]:
        if self.index is None or not self.ready or self._semantic_unavailable:
            return []
        from .embeddings import encode_texts
        try:
            query_vector = encode_texts(["query: " + query], self.model_name)
        except ValueError as exc:
            # An oversized query can still use lexical retrieval. Do not
            # disable semantic retrieval for the next, shorter question.
            warnings.warn(f"Semantic query rejected ({exc}); using BM25", stacklevel=2)
            return []
        except (ImportError, OSError, RuntimeError) as exc:
            self._semantic_unavailable = True
            warnings.warn(f"Semantic model unavailable ({exc}); using BM25", stacklevel=2)
            return []
        if query_vector.shape[1] != self.index.d:
            raise RuntimeError("FAISS/query vector dimension mismatch")
        search_count = min(len(self.metadata), max(candidate_count * 2, candidate_count))
        while True:
            _, row_ids = self.index.search(query_vector, search_count)
            results = [int(raw_id) for raw_id in row_ids[0]
                       if 0 <= int(raw_id) < len(self.metadata)
                       and _compatible(self.metadata[int(raw_id)], query)]
            if len(results) >= candidate_count or search_count >= len(self.metadata):
                return results[:candidate_count]
            search_count = min(len(self.metadata), search_count * 2)



    def _fuse(self, semantic: list[int], bm25: list[int]) -> list[int]:
        ranks: dict[int, float] = defaultdict(float)
        for result_list in (semantic, bm25):
            for position, row_id in enumerate(dict.fromkeys(result_list), start=1):
                ranks[row_id] += 1.0 / (self.rrf_k + position)
        return sorted(ranks, key=lambda idx: (-ranks[idx], idx))

    def _rerank(self, query: str, row_ids: list[int]) -> list[int]:
        self._reranking_succeeded = False
        if (not row_ids or not self.enable_reranking
                or self._reranking_unavailable):
            return row_ids
        try:
            model = _reranker(self.reranker_model)
            # Stored chunk text has bounded context; full metadata can contain a
            # very long title/question and should not push the provision out.
            pairs = [(query, self.metadata[idx]["text"]) for idx in row_ids]
            raw_scores = model.predict(
                pairs, batch_size=self.reranker_batch_size,
                show_progress_bar=False,
            )
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            if not isinstance(raw_scores, (list, tuple)):
                raw_scores = [raw_scores]
            values = []
            for raw in raw_scores:
                if isinstance(raw, (list, tuple)):
                    if len(raw) != 1:
                        raise RuntimeError("Unexpected multi-class reranker output")
                    raw = raw[0]
                # Supports model outputs shaped [N] or [N,1].
                try:
                    score = float(raw)
                except (TypeError, ValueError):
                    raise RuntimeError("Unexpected multi-class reranker output")
                if not math.isfinite(score):
                    raise RuntimeError("Reranker returned a non-finite score")
                values.append(score)
            if len(values) != len(row_ids):
                raise RuntimeError("Reranker returned unexpected score count")
            self._reranking_succeeded = True
            return [row_ids[i] for i in sorted(
                range(len(row_ids)), key=lambda i: (-values[i], i)
            )]
        except Exception as exc:
            # Failure must not break the user-facing assistant; a visible
            # warning makes BM25+FAISS fallback explicit for debugging.
            self._reranking_unavailable = True
            warnings.warn(
                f"Cross-encoder unavailable ({type(exc).__name__}: {exc}); "
                "using hybrid RRF order", stacklevel=2
            )
            return row_ids

    def search(self, query: str) -> list[RetrievedChunk]:
        """Return context passages ordered by hybrid relevance, with C1..Ck IDs.

        `score` is a *relative rank-fusion score*, not a calibrated
        probability of legal correctness/applicability.
        """
        query = str(query or "").strip()
        if not query or not self.metadata:
            return []

        semantic = self._semantic_search(query, self.top_k_candidates)
        bm25 = self._bm25_search(query, self.bm25_top_k)
        fused = self._fuse(semantic, bm25)
        if not fused:
            return []

        shortlist = fused[:self.rerank_top_n]
        ordered = self._rerank(query, shortlist)

        # Lexical overlap is useful evidence, not a requirement for semantic
        # equivalence. A cross-language query may have no shared surface words.
        query_terms = set(_query_terms(query, self.enable_query_expansion))
        cross_language_query = any(char.isalpha() and ord(char) > 127 for char in query)
        if self._reranking_succeeded or cross_language_query or not bm25:
            gated = ordered
        else:
            gated = [idx for idx in ordered if self._doc_term_sets[idx] & query_terms]

        # Apply the wage guard only to wage-payment disputes. Salary-account,
        # tax and other salary questions must not be forced into labour law.
        original_terms = set(_tokens(query))
        salary_intent = bool(original_terms & {
            "salary", "salaries", "wage", "wages", "paycheck", "payroll",
            "remuneration", "वेतन", "तनख्वाह", "सैलरी",
        })
        payment_dispute = bool(original_terms & {
            "unpaid", "nonpayment", "delayed", "withheld", "arrears", "बकाया", "रुका", "नहीं",
        }) or bool(re.search(r"\b(?:not paid|non[- ]payment|salary delay)\b", query, re.I))
        if salary_intent and payment_dispute and not self._reranking_succeeded and not cross_language_query:
            wage_anchors = {"wage", "wages", "salary", "salaries", "remuneration",
                            "arrears", "payroll", "earnings", "वेतन", "तनख्वाह", "सैलरी"}
            gated = [idx for idx in gated if self._doc_term_sets[idx] & wage_anchors]

        # Remove exact repeats only. Near-identical legal provisions can differ
        # by a crucial exception or a single negation and must be retained.
        unique: list[int] = []
        fingerprints: set[str] = set()
        for idx in gated:
            row = self.metadata[idx]
            content = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
            fingerprint = json.dumps([row.get("jurisdiction"), row.get("document_title"),
                                      row.get("section"), row.get("question"),
                                      row.get("effective_date"), content], ensure_ascii=False)
            if fingerprint not in fingerprints:
                fingerprints.add(fingerprint)
                unique.append(idx)
        gated = unique


        if not gated:
            return []

        # Diversity: at most 3 passages per Act/guide unless necessary to
        # fill context. Prevent one noisy source from monopolizing citations.
        counts: Counter[str] = Counter()
        selected: list[int] = []
        for idx in gated:
            doc = str(self.metadata[idx].get("document_id") or idx)
            if counts[doc] >= 3:
                continue
            selected.append(idx)
            counts[doc] += 1
            if len(selected) == self.top_k_context:
                break
        if len(selected) < self.top_k_context:
            selected_set = set(selected)
            selected.extend([idx for idx in gated
                             if idx not in selected_set][:self.top_k_context - len(selected)])

        # Scores preserve the actual fused ranks; reranking determines order.
        fusion_scores: dict[int, float] = defaultdict(float)
        for ranking in (semantic, bm25):
            for rank, idx in enumerate(ranking, 1):
                fusion_scores[idx] += 1 / (self.rrf_k + rank)

        return [RetrievedChunk(
            citation_id=f"C{rank}",
            score=float(fusion_scores[idx]),
            text=str(self.metadata[idx].get("text") or ""),
            metadata=dict(self.metadata[idx]),
        ) for rank, idx in enumerate(selected, start=1)]


def _sha256(path: Path) -> str:
    return sha256_file(path)
