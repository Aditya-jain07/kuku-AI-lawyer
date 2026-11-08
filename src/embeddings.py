"""CPU E5 embeddings and verified, row-aligned FAISS index artifacts.

Chunk text already contains document/section context. Add E5 prefixes once.
Rebuilds are serialized, and failed replacements restore the previous files.
Readers must use load_faiss_index() to detect incomplete or changed artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata as package_metadata
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .utils import dump_json, read_jsonl, sha256_file, write_jsonl


LOGGER = logging.getLogger(__name__)
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
INDEX_FILES = ("index.faiss", "metadata.jsonl", "metadata.json", "index_manifest.json")
_BUILD_LOCK = ".index-build.lock"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Reuse a CPU model; never execute model-provided Python code."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is missing; install requirements.txt") from exc
    return SentenceTransformer(model_name, device="cpu", trust_remote_code=False)


def _embedding_dimension(model: Any) -> int:
    try:
        return _positive_int(model.get_sentence_embedding_dimension(), "model dimension")
    except ValueError as exc:
        raise RuntimeError("Embedding model did not report a valid dimension") from exc


def _model_token_limit(model: Any) -> int:
    if getattr(model, "tokenizer", None) is None:
        raise RuntimeError("Embedding model tokenizer is unavailable")
    try:
        limit = _positive_int(getattr(model, "max_seq_length", None), "model token limit")
    except ValueError as exc:
        raise RuntimeError("Embedding model token limit is unavailable") from exc
    declared = getattr(model.tokenizer, "model_max_length", None)
    if isinstance(declared, Integral) and not isinstance(declared, bool) and 0 < declared < 1_000_000:
        limit = min(limit, int(declared))
    return limit


def _validated_inputs(texts: list[str]) -> None:
    if not isinstance(texts, list):
        raise TypeError("Embedding inputs must be a list of strings")
    for position, text in enumerate(texts, start=1):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Embedding input {position} must be a nonempty string")
        prefix = next((item for item in (PASSAGE_PREFIX, QUERY_PREFIX) if text.startswith(item)), None)
        if prefix is None or not text[len(prefix):].strip():
            raise ValueError(f"Embedding input {position} requires 'query: ' or 'passage: ' and nonempty text")


def _input_lengths(texts: list[str], model: Any, labels: list[str] | None = None) -> list[int]:
    """Count complete inputs; queries and passages are never silently cut."""
    limit = _model_token_limit(model)
    if labels is not None and len(labels) != len(texts):
        raise ValueError("Token-validation labels and input counts differ")
    counts: list[int] = []
    oversized: list[tuple[str, int]] = []
    for position, text in enumerate(texts):
        count = len(model.tokenizer.encode(
            text, add_special_tokens=True, truncation=False, verbose=False,
        ))
        counts.append(count)
        if count > limit:
            oversized.append((labels[position] if labels else f"input {position + 1}", count))
    if oversized:
        raise ValueError(
            f"{len(oversized)} embedding inputs exceed the model limit of {limit} tokens. "
            f"First examples: {oversized[:5]}. Rechunk passages or shorten the retrieval query."
        )
    return counts


def _normalized_vectors(value: Any, count: int, dimension: int) -> np.ndarray:
    vectors = np.array(value, dtype=np.float32, order="C", copy=True)
    if vectors.shape != (count, dimension):
        raise RuntimeError(f"Unexpected embedding shape {vectors.shape}; expected ({count}, {dimension})")
    if not np.isfinite(vectors).all():
        raise RuntimeError("Embedding model produced NaN or infinite values")
    # Float64 accumulation avoids overflow/underflow in float32 norm arithmetic.
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise RuntimeError("Embedding model produced an invalid or zero-length vector")
    vectors /= norms[:, None]
    actual_norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(actual_norms, 1.0, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Embedding normalization failed")
    return vectors


def encode_texts(texts: list[str], model_name: str, *, batch_size: int = 32) -> np.ndarray:
    """Encode already-prefixed queries/passages as normalized float32 vectors.

    Existing calls encode_texts(['query: ...'], model_name) are preserved.
    An explicit empty prompt prevents a saved default prompt adding a second
    prefix. Token validation runs before inference for both retrieval and build.
    """
    batch_size = _positive_int(batch_size, "batch_size")
    _validated_inputs(texts)
    model = _load_model(model_name)
    dimension = _embedding_dimension(model)
    if not texts:
        return np.empty((0, dimension), dtype=np.float32)
    _input_lengths(texts, model)
    vectors = model.encode(
        texts, batch_size=batch_size, prompt="", precision="float32",
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    )
    return _normalized_vectors(vectors, len(texts), dimension)


def _check_token_lengths(chunks: list[dict[str, Any]], passages: list[str], model: Any) -> int:
    """Preserve the original validation helper and reject mismatched inputs."""
    if len(chunks) != len(passages):
        raise ValueError("Chunk and passage counts differ")
    counts = _input_lengths(
        passages, model, [str(chunk.get("chunk_id", "unknown")) for chunk in chunks],
    )
    for chunk, count in zip(chunks, counts):
        if "token_count" not in chunk:
            continue
        recorded = chunk["token_count"]
        if type(recorded) is not int or recorded != count:
            raise ValueError(
                f"Chunk {chunk['chunk_id']} records {recorded!r} tokens but the embedding "
                f"tokenizer counted {count}. Rebuild chunks with the configured model."
            )
    return _model_token_limit(model)


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _validate_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    ids: set[str] = set()
    passages: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk row {position} must be a dictionary")
        chunk_id, body = chunk.get("chunk_id"), chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Missing chunk_id at row {position}")
        if chunk_id in ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        if not isinstance(body, str) or not body.strip():
            raise ValueError(f"Empty chunk text: {chunk_id}")
        if _has_e5_prefix(body):
            raise ValueError(f"Chunk already has an E5 input prefix: {chunk_id}")
        ids.add(chunk_id)
        passages.append(PASSAGE_PREFIX + body)
    return passages


def _has_e5_prefix(text: str) -> bool:
    """Detect already-prepared inputs without altering stored source text."""
    lowered = text.lstrip().lower()
    return lowered.startswith((PASSAGE_PREFIX, QUERY_PREFIX))


@contextmanager
def _index_build_lock(destination: Path) -> Iterator[None]:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / _BUILD_LOCK
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Index rebuild already locked: {path}. If a previous process crashed, "
            "confirm it has stopped before removing the stale lock."
        ) from exc
    identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}))
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            current = path.stat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOGGER.warning("Could not remove index build lock %s: %s", path, exc)


def _remove_backup(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        LOGGER.warning("Index files are consistent; could not remove backup %s: %s", path, exc)


def _install_index_files(staging: Path, destination: Path) -> None:
    """Install a generation, manifest last; restore files on replacement errors.

    Caller holds the build lock. Backups live outside staging so a failed
    rollback cannot destroy the recovery copy when TemporaryDirectory exits.
    This is exception recovery, not an atomic multi-file/power-loss transaction.
    """
    staging, destination = Path(staging), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in INDEX_FILES:
        if not (staging / filename).is_file():
            raise FileNotFoundError(f"Missing staged artifact: {filename}")
    backup = Path(tempfile.mkdtemp(prefix=".kuku-index-recovery-", dir=destination))
    previous_files: set[str] = set()
    attempted: list[str] = []
    try:
        # Copy all old artifacts before replacing any of them.
        for filename in INDEX_FILES:
            previous = destination / filename
            if previous.exists():
                if not previous.is_file():
                    raise ValueError(f"Index artifact is not a regular file: {previous}")
                shutil.copy2(previous, backup / filename)
                previous_files.add(filename)
        for filename in INDEX_FILES:
            # Track before replacement, including a cancellation during replace.
            attempted.append(filename)
            os.replace(staging / filename, destination / filename)
    except BaseException as exc:
        failures: list[str] = []
        for filename in reversed(attempted):
            try:
                if filename in previous_files:
                    os.replace(backup / filename, destination / filename)
                else:
                    (destination / filename).unlink(missing_ok=True)
            except OSError as rollback_error:
                failures.append(f"{filename}: {rollback_error}")
        if failures:
            raise RuntimeError(
                f"Index installation and rollback failed. Recovery files retained at {backup}. "
                f"Restore these files before loading the index. Details: {failures}"
            ) from exc
        _remove_backup(backup)
        raise
    else:
        _remove_backup(backup)


def _faiss_module():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is missing; install requirements.txt") from exc
    return faiss


def load_faiss_index(
    vector_store_dir: Path, model_name: str | None = None,
    embedding_dim: int | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Load a verified index, aligned metadata and manifest for Retriever.

    Validates all saved representations and detects a rebuild during loading.
    Existing schema-2 manifests without a metadata.json checksum remain
    supported by comparing that file's full contents against metadata.jsonl.
    """
    faiss = _faiss_module()
    directory = Path(vector_store_dir)
    if (directory / _BUILD_LOCK).exists():
        raise RuntimeError("Index is being rebuilt; load it after the build finishes")
    for filename in INDEX_FILES:
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Missing index artifact: {directory / filename}")
    manifest_path = directory / "index_manifest.json"
    snapshot = manifest_path.read_bytes()
    manifest = json.loads(snapshot.decode("utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("Unsupported index manifest; rebuild the index")
    if (manifest.get("metric") != "cosine_similarity"
            or manifest.get("normalized_embeddings") is not True
            or manifest.get("passage_prefix") != PASSAGE_PREFIX
            or manifest.get("query_prefix") != QUERY_PREFIX
            or manifest.get("indexed_text_field") != "text"):
        raise ValueError("Index embedding conventions do not match this retriever")
    dimension = _positive_int(manifest.get("embedding_dim"), "manifest embedding_dim")
    count = _positive_int(manifest.get("chunk_count"), "manifest chunk_count")
    if not isinstance(manifest.get("embedding_model"), str) or not manifest["embedding_model"].strip():
        raise ValueError("Manifest embedding_model is missing")
    if model_name is not None and model_name != manifest["embedding_model"]:
        raise ValueError("Configured embedding model differs from the saved index; rebuild it")
    if embedding_dim is not None and _positive_int(embedding_dim, "embedding_dim") != dimension:
        raise ValueError("Configured embedding dimension differs from the saved index")
    for filename, key in (("index.faiss", "index_sha256"), ("metadata.jsonl", "metadata_sha256"),
                          ("metadata.json", "metadata_json_sha256")):
        expected = manifest.get(key)
        if key == "metadata_json_sha256" and expected is None:
            continue
        if not isinstance(expected, str) or _file_sha256(directory / filename) != expected:
            raise ValueError(f"Index artifact checksum failed: {filename}")
    metadata = read_jsonl(directory / "metadata.jsonl", missing_ok=False)
    _validate_chunks(metadata)
    compatibility_metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
    if compatibility_metadata != metadata:
        raise ValueError("metadata.json and metadata.jsonl do not contain identical ordered records")
    index = faiss.read_index(str(directory / "index.faiss"))
    if (not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT
            or index.ntotal != count or index.d != dimension or len(metadata) != count):
        raise ValueError("FAISS index, manifest and metadata count/dimension/type do not agree")
    for start in range(0, count, 1024):
        vectors = index.reconstruct_n(start, min(1024, count - start))
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.isfinite(vectors).all() or not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("Saved FAISS vectors are not finite unit-length vectors")
    if (directory / _BUILD_LOCK).exists() or manifest_path.read_bytes() != snapshot:
        raise RuntimeError("Index changed during loading; retry after rebuilding finishes")
    return index, metadata, manifest


def _package_version(name: str) -> str | None:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def _model_revision(model: Any) -> str | None:
    try:
        config = model._first_module().auto_model.config
    except AttributeError:
        return None
    revision = getattr(config, "_commit_hash", None)
    return revision if isinstance(revision, str) and revision else None


def build_faiss_index(
    chunks_path: Path, vector_store_dir: Path, model_name: str,
    embedding_dim: int, *, batch_size: int = 32,
) -> dict[str, Any]:
    """Build an exact cosine-similarity index without reordering chunk rows.

    Keep the Streamlit app stopped during rebuilding: cached, in-memory
    retrievers remain on their previous generation until explicitly reloaded.
    """
    faiss = _faiss_module()
    embedding_dim = _positive_int(embedding_dim, "embedding_dim")
    batch_size = _positive_int(batch_size, "batch_size")
    chunks_path, vector_store_dir = Path(chunks_path), Path(vector_store_dir)
    if chunks_path.resolve() in {(vector_store_dir / name).resolve() for name in INDEX_FILES}:
        raise ValueError("Source chunks must not be an index output artifact")
    source_hash = _file_sha256(chunks_path)
    chunks = read_jsonl(chunks_path, missing_ok=False)
    if _file_sha256(chunks_path) != source_hash:
        raise RuntimeError("Source chunks changed while reading; retry with stable inputs")
    if not chunks:
        raise RuntimeError(f"No chunks found in {chunks_path}")
    passages = _validate_chunks(chunks)
    with _index_build_lock(vector_store_dir):
        model = _load_model(model_name)
        actual_dim = _embedding_dimension(model)
        if actual_dim != embedding_dim:
            raise ValueError(f"Configured dimension {embedding_dim} differs from model dimension {actual_dim}")
        model_limit = _check_token_lengths(chunks, passages, model)
        LOGGER.info("Embedding %s verified passages with %s on CPU", len(chunks), model_name)
        # Cache completed batches independently of staged index artifacts.
        # A retry after a save failure does not repeat successful inference.
        cache_identity = json.dumps([source_hash, model_name, _model_revision(model),
                                     embedding_dim, batch_size, model_limit,
                                     PASSAGE_PREFIX, "cache-v1"], sort_keys=True)
        import hashlib
        cache_key = hashlib.sha256(cache_identity.encode()).hexdigest()
        cache_dir = vector_store_dir / ".embedding-cache" / cache_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings = np.empty((len(passages), embedding_dim), dtype=np.float32)
        for start in range(0, len(passages), batch_size):
            stop = min(start + batch_size, len(passages))
            cache = cache_dir / f"batch-{start}.npy"
            receipt = cache.with_suffix(".json")
            vectors = None
            if cache.is_file() and receipt.is_file():
                try:
                    info = json.loads(receipt.read_text(encoding="utf-8"))
                    if info.get("sha256") == sha256_file(cache):
                        candidate = np.load(cache, allow_pickle=False)
                        norms = np.linalg.norm(candidate.astype(np.float64), axis=1)
                        if (candidate.shape == (stop - start, embedding_dim)
                                and candidate.dtype == np.float32
                                and np.isfinite(candidate).all()
                                and np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5)):
                            vectors = candidate
                except (OSError, ValueError, TypeError, AttributeError, EOFError):
                    LOGGER.warning("Ignoring invalid embedding cache batch %s", start)
            if vectors is None:
                vectors = encode_texts(passages[start:stop], model_name, batch_size=batch_size)
                temporary_cache = cache.with_suffix(".tmp")
                try:
                    with temporary_cache.open("wb") as handle:
                        np.save(handle, vectors, allow_pickle=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary_cache, cache)
                    dump_json(receipt, {"sha256": sha256_file(cache)})
                finally:
                    temporary_cache.unlink(missing_ok=True)
            embeddings[start:stop] = vectors
            if start == 0 or stop == len(passages) or (start // batch_size + 1) % 10 == 0:
                LOGGER.info("Embedding progress: %s/%s passages (completed batches cached)", stop, len(passages))
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)
        if index.ntotal != len(chunks) or index.d != embedding_dim:
            raise RuntimeError("FAISS insertion failed count/dimension checks")

        # Staging shares the destination filesystem, including mounted stores.
        with tempfile.TemporaryDirectory(prefix=".kuku-index-build-", dir=vector_store_dir) as temporary:
            staging = Path(temporary)
            faiss.write_index(index, str(staging / "index.faiss"))
            # Windows requires a writable descriptor for fsync/_commit.
            with (staging / "index.faiss").open("r+b") as handle:
                os.fsync(handle.fileno())
            write_jsonl(staging / "metadata.jsonl", chunks)
            dump_json(staging / "metadata.json", chunks)
            manifest = {
                "schema_version": 2, "generation_id": uuid.uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model": model_name, "embedding_model_revision": _model_revision(model),
                "embedding_dim": embedding_dim, "embedding_dtype": "float32",
                "metric": "cosine_similarity", "normalized_embeddings": True,
                "passage_prefix": PASSAGE_PREFIX, "query_prefix": QUERY_PREFIX,
                "indexed_text_field": "text", "chunk_count": len(chunks),
                "source_chunks": str(chunks_path.resolve()), "source_chunks_sha256": source_hash,
                "index_sha256": _file_sha256(staging / "index.faiss"),
                "metadata_sha256": _file_sha256(staging / "metadata.jsonl"),
                "metadata_json_sha256": _file_sha256(staging / "metadata.json"),
                "model_max_seq_length": model_limit,
                "sentence_transformers_version": _package_version("sentence-transformers"),
                "faiss_version": getattr(faiss, "__version__", None),
            }
            dump_json(staging / "index_manifest.json", manifest)
            reloaded, reloaded_metadata, _ = load_faiss_index(staging, model_name, embedding_dim)
            if reloaded_metadata != chunks:
                raise RuntimeError("Serialized metadata changed row order or contents")
            # Verify saved vector rows against the exact matrix that was added.
            for start in range(0, len(chunks), 1024):
                amount = min(1024, len(chunks) - start)
                stored = reloaded.reconstruct_n(start, amount)
                if not np.array_equal(stored, embeddings[start:start + amount]):
                    raise RuntimeError("Serialized FAISS vector rows changed")
            if _file_sha256(chunks_path) != source_hash:
                raise RuntimeError("Source chunks changed during indexing; previous index preserved")
            _install_index_files(staging, vector_store_dir)

    LOGGER.info("FAISS index created: %s vectors, %s dimensions", index.ntotal, index.d)
    return {
        "chunk_count": int(index.ntotal), "embedding_dim": int(index.d),
        "index_path": str(vector_store_dir / "index.faiss"),
        "manifest_path": str(vector_store_dir / "index_manifest.json"),
        "generation_id": manifest["generation_id"],
    }
