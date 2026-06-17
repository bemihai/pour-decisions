"""Interactive CLI for curating ground_truth_chunk_ids in the golden dataset.

Iterates through all ``rag_only`` samples that still have empty
``ground_truth_chunk_ids``, displays numbered chunk previews retrieved from
ChromaDB, and writes the selected IDs back to the JSONL file immediately after
each question is answered.

Progress is persisted after every question, so the session can be interrupted
and resumed safely — already-curated samples are skipped on the next run.

Usage::

    python -m src.eval.scripts.chunk_id_curator
    python -m src.eval.scripts.chunk_id_curator --top-k 8
    python -m src.eval.scripts.chunk_id_curator --redo          # re-curate all, including already-done
    python -m src.eval.scripts.chunk_id_curator --dataset src/eval/wine_qa_golden.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.eval.scripts.chunk_id_lookup import lookup_chunk_ids
from src.utils import get_config, logger

# ANSI color codes — disabled automatically when stdout is not a tty.
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def red(t: str) -> str:
    return _c("31", t)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all lines from a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        Ordered list of parsed JSON objects.
    """
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows back to a JSONL file atomically via a temp-file rename.

    Writing to a temporary file and renaming ensures the file is never in a
    partially-written state if the process is interrupted mid-write.

    Args:
        path: Destination file path.
        rows: Data to serialise.
    """
    tmp_path = path.with_suffix(".jsonl.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on error; re-raise so the caller can log it.
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _patch_chunk_ids(rows: list[dict[str, Any]], sample_id: str, chunk_ids: list[str]) -> None:
    """Update ground_truth_chunk_ids for one sample in-place.

    Args:
        rows: All JSONL rows, modified in-place.
        sample_id: The ``id`` field to target.
        chunk_ids: New chunk ID list to write.
    """
    for row in rows:
        if row.get("id") == sample_id:
            row["ground_truth_chunk_ids"] = chunk_ids
            return


def _print_header(index: int, total: int, sample: dict[str, Any]) -> None:
    """Print the per-question banner.

    Args:
        index: 1-based position in the curation session.
        total: Total number of questions to curate.
        sample: Raw JSONL row for the current question.
    """
    difficulty = sample.get("difficulty", "?")
    tags = ", ".join(sample.get("tags") or [])
    existing = sample.get("ground_truth_chunk_ids") or []
    existing_note = dim(f"  (already has {len(existing)} chunk ID(s))") if existing else ""

    print()
    print(bold("─" * 70))
    print(
        bold(f"  Question {index}/{total}")
        + "  " + cyan(sample["id"]) + "  "
        + dim(f"{difficulty} | {tags}")
        + existing_note
    )
    print(bold("─" * 70))
    print()
    print(f"  {bold('Q:')} {sample['question']}")
    print()
    print(dim(f"  Ground truth: {sample.get('ground_truth', '')}"))
    print()


def _print_candidates(candidates: list[dict[str, Any]]) -> None:
    """Print numbered candidate chunk previews.

    Args:
        candidates: Output of :func:`~src.eval.scripts.chunk_id_lookup.lookup_chunk_ids`.
    """
    for c in candidates:
        sim = c.get("similarity")
        sim_str = f"{sim:.4f}" if sim is not None else "  n/a"
        source = (c.get("source") or "unknown")[-60:]
        preview = c.get("preview") or ""
        rank_label = yellow(f"[{c['rank']}]")
        sim_label = green(f"sim={sim_str}")

        print(f"  {rank_label}  {sim_label}  {dim(source)}")
        print(f"       {preview}")
        print()


def _parse_selection(raw: str, max_rank: int) -> list[int] | None:
    """Parse the user's chunk selection into a list of 1-based ranks.

    Args:
        raw: Raw input string from the user.
        max_rank: Highest valid rank number.

    Returns:
        List of selected 1-based ranks, or ``None`` to signal skip.
    """
    raw = raw.strip().lower()

    if raw in {"", "s", "skip"}:
        return None

    if raw in {"a", "all"}:
        return list(range(1, max_rank + 1))

    ranks: list[int] = []
    for token in raw.replace(",", " ").split():
        try:
            rank = int(token)
        except ValueError:
            print(red(f"  Warning: '{token}' is not a valid number — skipped."))
            continue
        if rank < 1 or rank > max_rank:
            print(red(f"  Warning: {rank} is out of range (1-{max_rank}) — skipped."))
            continue
        if rank not in ranks:
            ranks.append(rank)

    return sorted(ranks)


def run_curation(
    dataset_path: Path,
    top_k: int = 10,
    redo: bool = False,
    category: str = "rag_only",
) -> None:
    """Run the interactive chunk ID curation session.

    Args:
        dataset_path: Path to the golden dataset JSONL file.
        top_k: Number of candidate chunks to retrieve per question.
        redo: When True, re-curate samples that already have chunk IDs.
        category: Only curate samples from this category.
    """
    rows = _load_jsonl(dataset_path)

    all_category = [r for r in rows if r.get("category") == category]
    pending = [
        row for row in all_category
        if redo or not row.get("ground_truth_chunk_ids")
    ]
    already_done = [r for r in all_category if r.get("ground_truth_chunk_ids")]

    total = len(pending)
    if total == 0:
        print(green(f"\nAll {category} samples already have chunk IDs. Use --redo to re-curate.\n"))
        return

    print()
    print(bold("  Pour Decisions — Chunk ID Curator"))
    print(dim(f"  Dataset : {dataset_path}"))
    print(dim(f"  Category: {category}"))
    print(dim(f"  Top-k   : {top_k}  |  ChromaDB retrieval, no LLM calls"))
    print()

    if already_done:
        print(green(f"  Already curated ({len(already_done)}/{len(all_category)}):"))
        for r in already_done:
            ids = r.get("ground_truth_chunk_ids", [])
            print(dim(f"    {r['id']}  ({len(ids)} chunk ID(s))"))
        print()

    print(yellow(f"  Remaining: {total} question(s) with no chunk IDs"))
    print(dim("  Progress is saved to disk after every question."))
    print(dim("  Commands: enter numbers (e.g. '1 3 5'), 'all', or press Enter to skip"))
    print()

    curated = 0
    for idx, sample in enumerate(pending, start=1):
        _print_header(index=idx, total=total, sample=sample)

        try:
            candidates = lookup_chunk_ids(question=sample["question"], top_k=top_k)
        except Exception as exc:
            print(red(f"  Retrieval failed: {exc}"))
            print(dim("  Skipping this sample."))
            continue

        if not candidates:
            print(red("  No chunks returned by retriever. Skipping."))
            continue

        _print_candidates(candidates)

        while True:
            try:
                raw = input(cyan("  Enter chunk numbers to keep (or Enter to skip): "))
            except (EOFError, KeyboardInterrupt):
                print()
                done_so_far = sum(
                    1 for r in rows
                    if r.get("category") == category and r.get("ground_truth_chunk_ids")
                )
                print(yellow(
                    f"\n  Session interrupted. {curated} question(s) saved in this session "
                    f"({done_so_far}/{len(all_category)} total curated)."
                ))
                print(dim("  Run again to continue — already-curated questions will be skipped.\n"))
                return

            ranks = _parse_selection(raw, max_rank=len(candidates))

            if ranks is None:
                print(dim("  Skipped (no chunk IDs saved; will appear again on next run).\n"))
                break

            if not ranks:
                print(red("  No valid numbers entered. Try again or press Enter to skip."))
                continue

            selected_ids = [candidates[r - 1]["chunk_id"] for r in ranks]
            _patch_chunk_ids(rows, sample["id"], selected_ids)
            _save_jsonl(dataset_path, rows)
            curated += 1

            remaining = total - idx  # questions still ahead after this one
            print(green(f"  Saved {len(selected_ids)} chunk ID(s) for {sample['id']} — written to disk."))
            for chunk_id in selected_ids:
                print(dim(f"    - {chunk_id}"))
            if remaining > 0:
                print(dim(f"  {remaining} question(s) remaining. Run again at any time to continue."))
            print()
            break

    print(bold(f"\n  Done. Curated {curated}/{total} questions.\n"))


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    cfg = get_config()
    project_root = Path(__file__).resolve().parents[2]
    default_dataset = (project_root / str(cfg.eval.dataset_path)).resolve()

    parser = argparse.ArgumentParser(
        description="Interactively curate ground_truth_chunk_ids in the golden dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help=f"Path to the golden JSONL dataset (default: {default_dataset})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidate chunks to show per question (default: 10)",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        default=False,
        help="Re-curate samples that already have chunk IDs",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error("Dataset file not found: %s", args.dataset)
        return 1

    run_curation(dataset_path=args.dataset, top_k=args.top_k, redo=args.redo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
