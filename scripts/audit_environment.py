#!/usr/bin/env python3
"""Audit the current runtime environment. No installs, no downloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion_ml.data_engine.audit import audit_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit runtime environment")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA unavailable")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    report = audit_environment(require_cuda=args.require_cuda)
    print(report)

    if args.out:
        path = report.save(args.out)
        print(f"\nReport saved to: {path}")

    if report.status in ("FAIL", "BLOCKED"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
