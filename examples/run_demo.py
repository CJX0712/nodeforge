"""End-to-end demo: runs the full benchmark and writes benchmark.json.

Usage:  python examples/run_demo.py
Result: deterministic benchmark.json at the repo root (same seed -> bitwise
identical core metrics).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodeforge.core.config import NodeForgeConfig
from nodeforge.eval.report import cli_table
from nodeforge.pipeline.pipeline import NodeForgePipeline


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cfg = NodeForgeConfig.from_env()
    pipe = NodeForgePipeline(cfg)
    report = pipe.benchmark()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(cli_table(report["results"]))
    gate = report["gate"]
    print("-" * 68)
    print(f"gate: mean_delta={gate['mean_delta']:.4f} "
          f"(threshold {gate['threshold']}), "
          f"significant_tiers={gate['significant_tiers']}, "
          f"passed={gate['passed']}")
    print(f"saved -> {out}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
