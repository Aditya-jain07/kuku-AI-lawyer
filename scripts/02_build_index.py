"""Phase 2: verify extracted records, chunk documents and build FAISS.

Run phases sequentially. Stop the Streamlit app before rebuilding its index.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# Support direct execution from any working directory, without PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="CPU embedding batch size; reduce it if memory is limited (default: 32).")
    parser.add_argument("--require-complete", action="store_true",
                        help="Reject a partial Phase 1 extraction.")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        from config import settings
        from src.chunking import build_chunks
        from src.embeddings import _index_build_lock, build_faiss_index
        from src.utils import read_jsonl, sha256_file, write_jsonl

        extracted_path = settings.processed_dir / "extracted.jsonl"
        manifest_path = settings.root_dir / "data" / "manifest.json"
        if not extracted_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Phase 1 output or manifest is missing. Run: python scripts/01_ingest.py")
        manifest_snapshot = manifest_path.read_bytes()
        manifest = json.loads(manifest_snapshot.decode("utf-8-sig"))
        if not isinstance(manifest, dict):
            raise ValueError("The Phase 1 manifest must be a JSON object")
        expected_hash = manifest.get("extracted_sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise RuntimeError("The Phase 1 manifest has no extraction checksum. Rerun the updated scripts/01_ingest.py.")
        if sha256_file(extracted_path) != expected_hash:
            raise RuntimeError("Extracted records do not match the Phase 1 manifest. Rerun Phase 1 before indexing.")
        records = read_jsonl(extracted_path, missing_ok=False)
        if not records:
            raise RuntimeError("No extracted records found. Run: python scripts/01_ingest.py")
        if type(manifest.get("record_count")) is not int or manifest["record_count"] != len(records):
            raise RuntimeError("Extracted record count differs from the Phase 1 manifest")
        if type(manifest.get("partial")) is not bool:
            raise ValueError("Phase 1 manifest must explicitly declare partial as true/false")
        if args.require_complete and manifest["partial"]:
            raise ValueError("Complete extraction required; rerun Phase 1 without failed sources")
        if manifest.get("partial"):
            logging.warning("Phase 1 published a partial corpus; this index covers only its available documents.")
        # Reject stale extraction after adding, moving, deleting or editing PDFs.
        inventory = manifest.get("sources")
        if not isinstance(inventory, list) or not inventory:
            raise ValueError("Phase 1 source inventory is missing")
        known = {}
        raw_root = settings.raw_dir.resolve()
        for source in inventory:
            if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                raise ValueError("Invalid Phase 1 source inventory entry")
            relative = source["path"]
            candidate = (raw_root / relative).resolve()
            if not candidate.is_relative_to(raw_root) or relative in known:
                raise ValueError("Invalid or duplicate raw-relative source path")
            known[relative] = candidate
            if not candidate.is_file():
                raise RuntimeError(f"Source missing since extraction: {relative}; rerun Phase 1")
            expected = source.get("sha256")
            # Failed unreadable sources can have no hash in an explicitly partial corpus.
            if source.get("status") in {"ok", "warning", "duplicate"} and (
                not isinstance(expected, str) or not expected
            ):
                raise ValueError(f"Missing source checksum: {relative}; rerun Phase 1")
            if expected and sha256_file(candidate) != expected:
                raise RuntimeError(f"Source changed since extraction: {relative}; rerun Phase 1")
        current = {p.relative_to(settings.raw_dir).as_posix()
                   for p in settings.raw_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}}
        if current != set(known):
            raise RuntimeError("Raw document collection changed; rerun Phase 1 before indexing")
        print(f"Loaded {len(records)} checksum-verified extracted records", flush=True)

        chunks = build_chunks(records, settings.chunk_target_tokens, settings.chunk_overlap_tokens)
        if not chunks:
            raise RuntimeError("Chunking produced no records. Review the extracted documents and chunking settings.")
        chunks_path = settings.processed_dir / "chunks.jsonl"
        # The vector-store lock starts inside build_faiss_index. Also serialize
        # the chunk write and recovery in processed/ across Phase 2 runs.
        with _index_build_lock(chunks_path.parent):
            if (sha256_file(extracted_path) != expected_hash
                    or manifest_path.read_bytes() != manifest_snapshot):
                raise RuntimeError("Phase 1 output changed during chunking. Run the phases sequentially and retry.")
            recovery_dir = Path(tempfile.mkdtemp(prefix=".chunks-recovery-", dir=chunks_path.parent))
            backup = recovery_dir / chunks_path.name
            keep_backup = False
            try:
                had_chunks = chunks_path.exists()
                if had_chunks:
                    shutil.copy2(chunks_path, backup)
                try:
                    write_jsonl(chunks_path, chunks)
                    print(f"Created {len(chunks)} chunks", flush=True)
                    result = build_faiss_index(
                        chunks_path=chunks_path,
                        vector_store_dir=settings.vector_store_dir,
                        model_name=settings.embedding_model,
                        embedding_dim=settings.embedding_dim,
                        batch_size=args.batch_size,
                    )
                except BaseException:
                    try:
                        if had_chunks:
                            backup.replace(chunks_path)
                        else:
                            chunks_path.unlink(missing_ok=True)
                    except OSError:
                        keep_backup = had_chunks
                        if keep_backup:
                            logging.error("Previous chunks may be recovered from %s", backup)
                        raise
                    raise
            finally:
                if not keep_backup:
                    try:
                        shutil.rmtree(recovery_dir)
                    except OSError as exc:
                        logging.warning("Could not remove chunk recovery directory: %s", exc)
    except ModuleNotFoundError as exc:
        traceback.print_exc()
        print(f"Missing module: {exc.name}. Check the project layout and installed requirements.", file=sys.stderr)
        return 1
    except Exception as exc:
        # Include the failing source line, not only a generic errno.
        traceback.print_exc()
        print(f"Phase 2 failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Phase 2 interrupted.", file=sys.stderr)
        return 130

    print("\nPhase 2 complete")
    print(f"Extracted records: {len(records)}")
    print(f"Indexed chunks: {result['chunk_count']}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Embedding dimensions: {result['embedding_dim']}")
    print(f"Index path: {result['index_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
