"""Phase 3: validate corpus integrity, source provenance and FAISS alignment.

These checks do not measure retrieval quality or verify legal correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

# Support direct execution from any working directory, without PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {label}")
    return value


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true",
                        help="Fail if Phase 1 published a partial corpus.")
    parser.add_argument("--verify-tokens", action="store_true",
                        help="Recount complete E5 inputs using the actual embedding tokenizer (may download it).")
    args = parser.parse_args(argv)
    try:
        from config import settings
        from src.embeddings import load_faiss_index
        from src.utils import read_jsonl, sha256_file

        extracted_path = settings.processed_dir / "extracted.jsonl"
        chunks_path = settings.processed_dir / "chunks.jsonl"
        manifest_path = settings.root_dir / "data" / "manifest.json"
        manifest_snapshot = manifest_path.read_bytes()
        manifest = json.loads(manifest_snapshot.decode("utf-8-sig"))
        if not isinstance(manifest, dict):
            raise ValueError("Phase 1 manifest must be an object")
        extracted_hash = sha256_file(extracted_path)
        if manifest.get("extracted_sha256") != extracted_hash:
            raise ValueError("Extraction checksum mismatch or missing checksum; rerun scripts/01_ingest.py")
        records = read_jsonl(extracted_path, missing_ok=False)
        if (not records or type(manifest.get("record_count")) is not int
                or manifest["record_count"] != len(records)):
            raise ValueError("Extracted records are empty or differ from the manifest count")

        record_map: dict[str, dict[str, Any]] = {}
        extracted_docs: set[str] = set()
        for record in records:
            record_id = _identifier(record.get("record_id"), "record_id")
            doc = _identifier(record.get("document_id"), "extracted document_id")
            if record_id in record_map:
                raise ValueError(f"Duplicate extracted record ID: {record_id}")
            if not isinstance(record.get("text"), str) or not record["text"].strip():
                raise ValueError(f"Empty extracted text: {record_id}")
            if record.get("source_type") == "pdf" and not _positive(record.get("page_number")):
                raise ValueError(f"Invalid PDF page number: {record_id}")
            record_map[record_id] = record
            extracted_docs.add(doc)
        print("\n=== DOCUMENT EXTRACTION ===")
        print(f"Extracted records: {len(records)}")
        print(f"Unique extracted documents: {len(extracted_docs)}")
        for kind, count in sorted(Counter(str(row.get("source_type", "unknown")) for row in records).items()):
            print(f"  {kind}: {count}")

        sources = manifest.get("sources")
        if not isinstance(sources, list) or any(not isinstance(source, dict) for source in sources):
            raise ValueError("Invalid source inventory in Phase 1 manifest")
        if (type(manifest.get("source_count")) is not int
                or manifest["source_count"] != len(sources)):
            raise ValueError("Source inventory count differs from the manifest")
        if type(manifest.get("partial")) is not bool:
            raise ValueError("Phase 1 manifest must explicitly declare partial as true/false")
        raw_root = settings.raw_dir.resolve()
        inventory_paths = set()
        canonical_docs = set()
        for source in sources:
            relative = _identifier(source.get("path"), "source path")
            path = (raw_root / relative).resolve()
            if not path.is_relative_to(raw_root) or relative in inventory_paths:
                raise ValueError("Invalid or duplicate source path in inventory")
            inventory_paths.add(relative)
            if not path.is_file():
                raise ValueError(f"Raw source missing: {relative}; rerun Phase 1")
            expected = source.get("sha256")
            if expected and sha256_file(path) != expected:
                raise ValueError(f"Raw source changed: {relative}; rerun Phase 1")
            if source.get("status") in {"ok", "warning"}:
                canonical_docs.add(_identifier(expected, "source checksum"))
        current_paths = {path.relative_to(settings.raw_dir).as_posix()
                         for path in settings.raw_dir.rglob("*")
                         if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}}
        if current_paths != inventory_paths:
            raise ValueError("Raw documents differ from the extraction inventory; rerun Phase 1")
        if canonical_docs != extracted_docs:
            raise ValueError("Extracted document IDs differ from canonical inventory checksums")
        statuses = Counter(str(source.get("status", "unknown")) for source in sources)
        print("\n=== SOURCE INVENTORY ===")
        print(f"Files discovered: {len(sources)}")
        for status, count in sorted(statuses.items()):
            print(f"  {status}: {count}")
        failed = [source.get("path", "unknown") for source in sources
                  if source.get("status") in {"error", "empty"}]
        if failed:
            print("Documents needing attention:")
            for path in failed[:20]:
                print(f"  - {path}")
        for source in sources:
            suspect_pages = [
                issue["page_number"] for issue in source.get("page_issues", [])
                if issue.get("status") == "suspect_text_encoding"
            ]
            if suspect_pages:
                pages = ", ".join(str(page) for page in suspect_pages[:12])
                remaining = f" (+{len(suspect_pages) - 12} more)" if len(suspect_pages) > 12 else ""
                print(f"  Suspect text encoding in {source['path']}, pages {pages}{remaining}")
        if manifest.get("partial"):
            print("WARNING: Phase 1 published a partial corpus; alignment does not establish source completeness.")
            if args.require_complete:
                raise ValueError("A complete corpus was required, but Phase 1 reports partial extraction")

        chunks_hash = sha256_file(chunks_path)
        chunks = read_jsonl(chunks_path, missing_ok=False)
        if not chunks:
            raise ValueError("No chunks found. Run: python scripts/02_build_index.py")
        ids: set[str] = set()
        chunk_docs: set[str] = set()
        covered_records: set[str] = set()
        pdf_count = 0
        for chunk in chunks:
            chunk_id = _identifier(chunk.get("chunk_id"), "chunk_id")
            doc = _identifier(chunk.get("document_id"), "chunk document_id")
            if chunk_id in ids:
                raise ValueError(f"Duplicate chunk ID: {chunk_id}")
            ids.add(chunk_id)
            chunk_docs.add(doc)
            if doc not in extracted_docs:
                raise ValueError(f"Chunk references an unknown document: {chunk_id}")
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError(f"Empty chunk text: {chunk_id}")
            origins = chunk.get("source_record_ids")
            if not isinstance(origins, list) or not origins or any(not isinstance(item, str) for item in origins):
                raise ValueError(f"Missing source_record_ids: {chunk_id}; rerun the updated Phase 2")
            if len(origins) != len(set(origins)):
                raise ValueError(f"Repeated source record ID within {chunk_id}")
            for origin in origins:
                record = record_map.get(origin)
                if record is None or record["document_id"] != doc:
                    raise ValueError(f"Invalid source record reference in {chunk_id}: {origin}")
                if record.get("source_type") != chunk.get("source_type"):
                    raise ValueError(f"Source-type mismatch in {chunk_id}")
                for field in ("relative_source_path", "document_title", "source_url",
                              "jurisdiction", "effective_date", "faq_number"):
                    if chunk.get(field) != record.get(field):
                        raise ValueError(f"Citation metadata mismatch in {chunk_id}: {field}")
            covered_records.update(origins)
            if chunk.get("source_type") == "pdf":
                pdf_count += 1
                pages = chunk.get("page_numbers")
                if (not isinstance(pages, list) or not pages or not all(_positive(page) for page in pages)
                        or pages != sorted(set(pages))
                        or any(right != left + 1 for left, right in zip(pages, pages[1:]))
                        or chunk.get("page_start") != pages[0]
                        or chunk.get("page_end") != pages[-1]
                        or chunk.get("page_number") != pages[0]):
                    raise ValueError(f"Invalid/nonconsecutive PDF page provenance: {chunk_id}")
                actual_pages = sorted({record_map[origin]["page_number"] for origin in origins})
                if actual_pages != pages:
                    raise ValueError(f"Cited pages differ from source records: {chunk_id}")
        if extracted_docs - chunk_docs:
            raise ValueError(f"{len(extracted_docs - chunk_docs)} extracted documents have no chunks")
        missing = record_map.keys() - covered_records
        if missing:
            raise ValueError(f"{len(missing)} extracted records have no chunk coverage; examples: {sorted(missing)[:5]}")
        print("\n=== CHUNK VALIDATION ===")
        print(f"Generated chunks: {len(chunks)}")
        print(f"PDF chunks with valid source-page references: {pdf_count}")
        print(f"Extracted records referenced by chunks: {len(covered_records)}")

        index_manifest_path = settings.vector_store_dir / "index_manifest.json"
        index_snapshot = index_manifest_path.read_bytes()
        index, indexed_metadata, index_manifest = load_faiss_index(
            settings.vector_store_dir, settings.embedding_model, settings.embedding_dim,
        )
        if index_manifest.get("source_chunks_sha256") != chunks_hash:
            raise ValueError("FAISS was built from a different chunks.jsonl; rerun Phase 2")
        if indexed_metadata != chunks:
            raise ValueError("Indexed metadata differs from chunks.jsonl in content or order")
        limit = index_manifest.get("model_max_seq_length")
        if not _positive(limit):
            raise ValueError("Index manifest has no valid model token limit")
        limit = min(limit, settings.embedding_max_tokens, settings.chunk_target_tokens)
        for chunk in chunks:
            count = chunk.get("token_count")
            if not _positive(count) or count > limit:
                raise ValueError(f"Invalid stored token count: {chunk['chunk_id']}")
        if args.verify_tokens:
            from src.chunking import get_tokenizer
            tokenizer = get_tokenizer()
            for chunk in chunks:
                actual = len(tokenizer.encode("passage: " + chunk["text"],
                             add_special_tokens=True, truncation=False, verbose=False))
                if actual != chunk["token_count"] or actual > limit:
                    raise ValueError(f"Actual token count mismatch/overflow: {chunk['chunk_id']}")
            print("Actual embedding-tokenizer counts verified.")
        print("\n=== FAISS VALIDATION ===")
        print(f"FAISS vectors: {index.ntotal}")
        print(f"Embedding dimensions: {index.d}")
        print(f"Embedding model: {settings.embedding_model}")

        faqs = [record for record in records if record.get("source_type") == "faq"]
        if any(not _positive(record.get("faq_number")) for record in faqs):
            raise ValueError("FAQ records contain invalid FAQ numbers")
        identities = {(record["document_id"], record["faq_number"]) for record in faqs}
        print("\n=== FAQ COVERAGE ===")
        print(f"FAQ records extracted: {len(faqs)}")
        print(f"Distinct document/FAQ-number pairs: {len(identities)}")
        print(f"Additional records sharing FAQ numbers: {len(faqs) - len(identities)}")

        if (sha256_file(extracted_path) != extracted_hash or sha256_file(chunks_path) != chunks_hash
                or manifest_path.read_bytes() != manifest_snapshot
                or index_manifest_path.read_bytes() != index_snapshot):
            raise RuntimeError("Artifacts changed during validation; finish other phases and retry")
    except ModuleNotFoundError as exc:
        traceback.print_exc()
        print(f"FAIL: Missing module {exc.name}. Check project layout and installed requirements.", file=sys.stderr)
        return 1
    except Exception as exc:
        # Include the failing source line, not only a generic errno.
        traceback.print_exc()
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Validation interrupted.", file=sys.stderr)
        return 130

    print("\nDATA INTEGRITY CHECK PASSED" + (" — PARTIAL CORPUS" if manifest.get("partial") else ""))
    print("Documents, chunks and FAISS metadata are aligned.")
    print("This does not verify legal accuracy or retrieval relevance.")
    if not args.verify_tokens:
        print("Stored token counts checked; use --verify-tokens to independently recount them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
