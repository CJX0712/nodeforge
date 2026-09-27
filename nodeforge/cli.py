"""CLI entry: python -m nodeforge.cli <benchmark|run|models|info>"""
from __future__ import annotations

import argparse
import json
import sys

from nodeforge.core.config import NodeForgeConfig
from nodeforge.core.errors import NodeForgeError
from nodeforge.graph.factory import AVAILABLE_MODELS
from nodeforge.pipeline.pipeline import NodeForgePipeline


def _utf8() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nodeforge", description="NodeForge: GNN node classification benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("benchmark", help="full benchmark (tiers x seeds x models)")
    b.add_argument("--out", default="benchmark.json")
    b.add_argument("--seeds", type=int, default=0, help="override n_seeds")

    r = sub.add_parser("run", help="single tier, one seed")
    r.add_argument("--tier", required=True)
    r.add_argument("--model", default=None, help="single model name (default: all)")
    r.add_argument("--seed", type=int, default=None)

    sub.add_parser("models", help="list available models")
    sub.add_parser("info", help="show resolved config")
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "models":
            for m in sorted(AVAILABLE_MODELS):
                print(f"{m:<12}")
            return 0
        if args.cmd == "info":
            cfg = NodeForgeConfig.from_env()
            for k in sorted(vars(cfg)):
                print(f"{k:<20}{getattr(cfg, k)}")
            return 0

        cfg = NodeForgeConfig.from_env()
        if args.cmd == "benchmark":
            if args.seeds:
                cfg.n_seeds = args.seeds
            pipe = NodeForgePipeline(cfg)
            report = pipe.benchmark()
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
            from nodeforge.eval.report import cli_table

            print(cli_table(report["results"]))
            gate = report["gate"]
            print("-" * 68)
            print(f"gate: mean_delta={gate['mean_delta']:.4f} "
                  f"(threshold {gate['threshold']}), "
                  f"significant_tiers={gate['significant_tiers']}, "
                  f"passed={gate['passed']}")
            print(f"saved -> {args.out} ({report['timing_sec']}s)")
            return 0

        if args.cmd == "run":
            pipe = NodeForgePipeline(cfg)
            seed = args.seed if args.seed is not None else cfg.seed
            models = [args.model] if args.model else None
            res = pipe.run_tier(args.tier, seed, models)
            header = f"{'model':<12}{'acc':<10}{'macro_f1':<10}"
            print(header)
            print("-" * len(header))
            for name in sorted(res):
                print(f"{name:<12}{res[name]['acc']:<10.4f}{res[name]['macro_f1']:<10.4f}")
            return 0
    except NodeForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
