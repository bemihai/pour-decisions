"""CLI entry point for running the eval harness."""
from omegaconf import DictConfig

import argparse
import asyncio
from pathlib import Path

from src.eval.dataset import filter_golden_samples, load_golden_dataset
from src.eval.models import CATEGORIES, DIFFICULTIES, EVAL_BACKENDS, EVAL_MODES
from src.eval.metrics import precision_at_k, reciprocal_rank
from src.eval.models import GoldenSample, SampleResult
from src.eval.preflight import run_preflight
from src.eval.reporter import EvalReporter
from src.eval.runner import EvalRunner
from src.utils import compute_file_hash, get_config, logger, parse_csv_arg


def _attach_retrieval_metrics(
    results_by_id: dict[str, SampleResult],
    samples: list[GoldenSample],
    k_values: list[int],
) -> None:
    """Attach local retrieval metrics to per-sample score maps.

    Args:
        results_by_id: Map from sample id to SampleResult.
        samples: Golden samples used to resolve relevant chunk IDs.
        k_values: Precision cutoffs to compute.
    """
    for sample in samples:
        result = results_by_id.get(sample.id)
        if (result is None) or (result.status != "passed"):
            continue

        relevant_ids = sample.ground_truth_chunk_ids
        retrieved_ids = result.retrieved_chunk_ids
        if not relevant_ids:
            continue

        result.scores["mrr"] = reciprocal_rank(
            retrieved_ids=retrieved_ids, 
            relevant_ids=relevant_ids
            )
        for k in k_values:
            result.scores[f"precision_at_{k}"] = precision_at_k(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=k,
            )


def build_parser(cfg: DictConfig) -> argparse.ArgumentParser:
    """Build the eval CLI parser."""
    eval_cfg = cfg.eval
    parser = argparse.ArgumentParser(description="Run Pour Decisions eval harness")

    parser.add_argument(
        "--mode",
        choices=list(EVAL_MODES),
        default=str(eval_cfg.default_mode)
    )
    parser.add_argument(
        "--backend",
        choices=list(EVAL_BACKENDS),
        default=str(eval_cfg.default_backend)
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated category filter"
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        default=None,
        help="Comma-separated difficulty filter"
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tag filter"
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Comma-separated sample id filter"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(str(eval_cfg.dataset_path))
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(str(eval_cfg.results_dir))
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=int(eval_cfg.max_concurrency)
    )
    parser.add_argument(
        "--push-to-phoenix",
        action="store_true",
        default=False,
        help="Push results to Phoenix as a named experiment (requires Phoenix running)",
    )
    parser.add_argument(
        "--phoenix-url",
        type=str,
        default=None,
        help="Phoenix base URL override",
    )
    return parser


def _validate_cli_filters(
    parser: argparse.ArgumentParser,
    dataset: list[GoldenSample],
    categories: list[str] | None,
    difficulties: list[str] | None,
    tags: list[str] | None,
    sample_ids: list[str] | None,
    validate_tag_filters: bool,
) -> None:
    """Validate CLI filter values against known dataset and schema values."""
    if categories:
        invalid_categories = sorted({value for value in categories if value not in CATEGORIES})
        if invalid_categories:
            parser.error(
                "Invalid categories: "
                f"{', '.join(invalid_categories)}. Valid categories: {', '.join(sorted(CATEGORIES))}."
            )

    if difficulties:
        invalid_difficulties = sorted({value for value in difficulties if value not in DIFFICULTIES})
        if invalid_difficulties:
            parser.error(
                "Invalid difficulties: "
                f"{', '.join(invalid_difficulties)}. Valid difficulties: {', '.join(sorted(DIFFICULTIES))}."
            )

    if sample_ids:
        valid_sample_ids = {sample.id for sample in dataset}
        invalid_sample_ids = sorted({value for value in sample_ids if value not in valid_sample_ids})
        if invalid_sample_ids:
            parser.error(
                "Invalid sample ids: "
                f"{', '.join(invalid_sample_ids)}. Valid sample ids: {', '.join(sorted(valid_sample_ids))}."
            )

    if not tags:
        return

    valid_tags = {tag for sample in dataset for tag in sample.tags}
    invalid_tags = sorted({value for value in tags if value not in valid_tags})
    if not invalid_tags:
        return

    if not validate_tag_filters:
        logger.warning(
            "Unknown tag filters requested: %s. Proceeding because eval.validate_tag_filters=false",
            ", ".join(invalid_tags),
        )
        return

    parser.error(
        "Invalid tags: "
        f"{', '.join(invalid_tags)}. Valid tags: {', '.join(sorted(valid_tags))}."
    )


def _build_run_metadata(
    dataset_path: Path,
    dataset: list[GoldenSample],
    selected_samples: list[GoldenSample],
    categories: list[str] | None,
    difficulties: list[str] | None,
    tags: list[str] | None,
    sample_ids: list[str] | None,
    args: argparse.Namespace,
    config: DictConfig,
    git_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build structured run metadata for reproducibility."""
    resolved_dataset_path = dataset_path.resolve()
    metadata: dict[str, object] = {
        "dataset": {
            "path": str(resolved_dataset_path),
            "content_hash": compute_file_hash(resolved_dataset_path),
            "total_sample_count": len(dataset),
            "selected_sample_count": len(selected_samples),
        },
        "filters": {
            "categories": categories,
            "difficulties": difficulties,
            "tags": tags,
            "sample_ids": sample_ids,
        },
        "execution": {
            "mode": args.mode,
            "backend": args.backend,
            "max_concurrency": args.max_concurrency,
            "sample_timeout_seconds": float(getattr(config.eval, "sample_timeout_seconds", 0) or 0),
            "push_to_phoenix": bool(args.push_to_phoenix),
            "phoenix_url": args.phoenix_url,
        },
    }
    if git_metadata:
        metadata["git"] = git_metadata
    return metadata


def main() -> int:
    """Run eval pipeline from CLI arguments."""
    config = get_config()
    parser = build_parser(config)
    args = parser.parse_args()

    categories = parse_csv_arg(args.categories)
    difficulties = parse_csv_arg(args.difficulties)
    tags = parse_csv_arg(args.tags)
    sample_ids = parse_csv_arg(args.sample_id)
    dataset = load_golden_dataset(args.dataset)
    _validate_cli_filters(
        parser=parser,
        dataset=dataset,
        categories=categories,
        difficulties=difficulties,
        tags=tags,
        sample_ids=sample_ids,
        validate_tag_filters=bool(getattr(config.eval, "validate_tag_filters", True)),
    )
    run_preflight(
        parser=parser,
        config=config,
        mode=args.mode,
        backend=args.backend,
    )

    # Load and filter golden samples based on CLI args
    samples = filter_golden_samples(
        dataset,
        categories=categories,
        difficulties=difficulties,
        tags=tags,
        sample_ids=sample_ids,
    )

    if not samples:
        logger.warning("No samples matched the selected filters. Exiting.")
        return 0

    # Run retrieval eval to get per-sample results
    runner = EvalRunner(
        backend=args.backend,
        config=config,
        generation_enabled=args.mode == "full",
    )
    run_metadata = _build_run_metadata(
        dataset_path=args.dataset,
        dataset=dataset,
        selected_samples=samples,
        categories=categories,
        difficulties=difficulties,
        tags=tags,
        sample_ids=sample_ids,
        args=args,
        config=config,
        git_metadata=runner.git_metadata,
    )
    results = asyncio.run(
        runner.run(samples=samples, max_concurrency=args.max_concurrency)
    )

    k_values = [int(k) for k in getattr(config.eval.retrieval_metrics, "k_values", [3, 5])]
    results_by_id = {result.id: result for result in results}
    _attach_retrieval_metrics(results_by_id=results_by_id, samples=samples, k_values=k_values)

    # full mode uses Ragas scoring with LLM-as-judge -> expensive
    if args.mode == "full":
        from src.eval.ragas_scorer import RagasScorer

        scorer = RagasScorer()
        scorer.score(results)
        if args.backend == "agent":
            scorer.score_agent_answers(results)

    # build and save the evaluation report
    reporter = EvalReporter()
    report = reporter.build(
        results=results,
        samples=samples,
        mode=args.mode,
        backend=args.backend,
        config_snapshot=runner.config_snapshot,
        git_sha=runner.git_sha,
        run_metadata=run_metadata,
    )
    output_path = reporter.save(report, output_dir=args.output_dir)
    reporter.print_summary(report)
    logger.info("Eval run completed: %s", output_path)

    if args.push_to_phoenix:
        from src.eval.phoenix_reporter import PhoenixReporter

        phoenix_reporter = PhoenixReporter(base_url=args.phoenix_url)
        experiment_url = phoenix_reporter.push(result=report, samples=samples)
        if experiment_url:
            logger.info("Phoenix experiment available at: %s", experiment_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
