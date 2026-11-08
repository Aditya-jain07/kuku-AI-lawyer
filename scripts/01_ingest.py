"""Phase 1: extract PDFs and FAQ text into processed/extracted.jsonl."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

# Support direct execution from any working directory, without PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Publish usable documents even if other sources fail; review the extraction reports.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        from config import settings
        from src.ingestion import ingest

        result = ingest(
            settings.raw_dir,
            settings.processed_dir,
            settings.root_dir / "data" / "manifest.json",
            allow_partial=args.allow_partial,
        )
    except ModuleNotFoundError as exc:
        traceback.print_exc()
        print(
            f"Missing module: {exc.name}. Check the project layout and installed requirements.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        # Include the failing source line, not only a generic errno.
        traceback.print_exc()
        print(f"Phase 1 failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Phase 1 interrupted.", file=sys.stderr)
        return 130

    if result.get("partial"):
        print("Phase 1 finished: a partial corpus was published. Review extraction_report.csv and errors.json.")
    else:
        print("Phase 1 complete")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
