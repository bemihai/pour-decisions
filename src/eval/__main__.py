"""CLI entry point for running the eval harness."""
from omegaconf import DictConfig

import argparse
import asyncio
from pathlib import Path

from src.eval.dataset import filter_golden_samples, load_golden_dataset
from src.eval.metrics import precision_at_k, reciprocal_rank
from src.eval.models import GoldenSample, SampleResult
from src.eval.phoenix_reporter import PhoenixReporter
from src.eval.ragas_scorer import RagasScorer
from src.eval.reporter import EvalReporter
from src.eval.runner import EvalRunner
from src.utils import get_config, logger, parse_csv_arg


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
        if (result is None) or (result.error is not None):
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
        choices=["retrieval", "full"],  
        default=str(eval_cfg.default_mode)
    )
    parser.add_argument(
        "--backend",
        choices=["rag", "agent"],
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


def main() -> int:
    """Run eval pipeline from CLI arguments."""
    config = get_config()
    parser = build_parser(config)
    args = parser.parse_args()

    # Load and filter golden samples based on CLI args
    samples = filter_golden_samples(
        load_golden_dataset(args.dataset),
        categories=parse_csv_arg(args.categories),
        difficulties=parse_csv_arg(args.difficulties),
        tags=parse_csv_arg(args.tags),
    )

    if not samples:
        logger.warning("No samples matched the selected filters. Exiting.")
        return 0

    # Run eval harness to get per-sample results
    runner = EvalRunner(backend=args.backend, config=config)
    results = asyncio.run(
        runner.run(samples=samples, mode=args.mode, max_concurrency=args.max_concurrency)
    )

    k_values = [int(k) for k in getattr(config.eval.retrieval_metrics, "k_values", [3, 5])]
    results_by_id = {result.id: result for result in results}
    _attach_retrieval_metrics(results_by_id=results_by_id, samples=samples, k_values=k_values)

    # full mode uses Ragas scoring with LLM-as-judge -> expensive
    if args.mode == "full":
        scorer = RagasScorer()
        scorer.score(results)

    reporter = EvalReporter()
    report = reporter.build(
        results=results,
        samples=samples,
        mode=args.mode,
        backend=args.backend,
        config_snapshot=runner.config_snapshot,
        git_sha=runner.git_sha,
    )
    output_path = reporter.save(report, output_dir=args.output_dir)
    reporter.print_summary(report)
    logger.info("Eval run completed: %s", output_path)

    if args.push_to_phoenix:
        phoenix_reporter = PhoenixReporter(base_url=args.phoenix_url)
        experiment_url = phoenix_reporter.push(result=report, samples=samples)
        if experiment_url:
            logger.info("Phoenix experiment available at: %s", experiment_url)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

