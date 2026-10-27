#!/usr/bin/env python3
"""
Validate data against a contract using synthetic or user-provided frames.
Does NOT download any dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sensor_fusion_ml.data_engine.contracts import (
    DataContract,
    Frame,
    Observation,
    SensorModality,
    Timestamp,
    TimeUnit,
)
from sensor_fusion_ml.data_engine.validator import DataValidator


def make_synthetic_frames(n: int = 20, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    frames = []
    t0 = 0.0
    for i in range(n):
        ts = Timestamp(value=t0 + i * 0.1, unit=TimeUnit.SECONDS)
        obs = {
            SensorModality.GNSS: Observation(
                modality=SensorModality.GNSS,
                timestamp=ts,
                data=rng.normal(size=3).astype(np.float64),
                is_valid=True,
            ),
            SensorModality.IMU: Observation(
                modality=SensorModality.IMU,
                timestamp=ts,
                data=rng.normal(size=6).astype(np.float64),
                is_valid=True,
            ),
        }
        frames.append(Frame(frame_id=f"f{i:04d}", timestamp=ts, observations=obs, sequence_id="seq00"))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate data (synthetic by default)")
    parser.add_argument("--out", type=str, default=None, help="JSON report path")
    parser.add_argument("--n-frames", type=int, default=20)
    args = parser.parse_args()

    contract = DataContract(
        name="synthetic_gnss_imu",
        required_modalities=[SensorModality.GNSS, SensorModality.IMU],
        time_unit=TimeUnit.SECONDS,
        min_frames=5,
        max_nan_ratio=0.0,
        require_finite=True,
        require_sorted_timestamps=True,
        require_unique_timestamps=True,
        max_time_gap_s=0.5,
        license="synthetic (no external data)",
        provenance="generated inside validate_data.py",
    )

    frames = make_synthetic_frames(n=args.n_frames)
    validator = DataValidator(contract)
    report = validator.validate_frames(frames, sequence_id="seq00")

    print(report)

    if args.out:
        path = report.save(args.out)
        print(f"\nReport saved to: {path}")

    return 0 if report.status in ("PASS", "PASS_WITH_WARNINGS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
