"""Unit tests for DataValidator – no external data."""

import numpy as np
import pytest

from sensor_fusion_ml.data_engine.contracts import (
    DataContract,
    Frame,
    Observation,
    SensorModality,
    Timestamp,
    TimeUnit,
)
from sensor_fusion_ml.data_engine.validator import DataValidator, ValidationStatus


def _make_frames(n=10, with_nan=False, shuffle=False):
    frames = []
    times = list(range(n))
    if shuffle:
        times = times[::-1]
    for i, t in enumerate(times):
        data = np.array([float(i), 0.0, 0.0])
        if with_nan and i == 3:
            data[0] = np.nan
        obs = {
            SensorModality.GNSS: Observation(
                SensorModality.GNSS,
                Timestamp(float(t) * 0.1),
                data,
                is_valid=True,
            )
        }
        frames.append(Frame(f"f{i}", Timestamp(float(t) * 0.1), obs, sequence_id="s0"))
    return frames


def test_pass_clean_data():
    contract = DataContract(
        name="clean",
        required_modalities=[SensorModality.GNSS],
        min_frames=5,
        require_finite=True,
        require_sorted_timestamps=True,
        require_unique_timestamps=True,
    )
    v = DataValidator(contract)
    report = v.validate_frames(_make_frames(10))
    assert report.status == ValidationStatus.PASS


def test_fail_missing_modality():
    contract = DataContract(
        name="need_imu",
        required_modalities=[SensorModality.GNSS, SensorModality.IMU],
        min_frames=1,
    )
    v = DataValidator(contract)
    report = v.validate_frames(_make_frames(5))
    assert report.status == ValidationStatus.FAIL
    assert any(c.name == "required_modalities" for c in report.checks)


def test_fail_nan():
    contract = DataContract(
        name="no_nan",
        required_modalities=[SensorModality.GNSS],
        min_frames=1,
        require_finite=True,
        max_nan_ratio=0.0,
    )
    v = DataValidator(contract)
    report = v.validate_frames(_make_frames(8, with_nan=True))
    assert report.status == ValidationStatus.FAIL
    finite_check = next(c for c in report.checks if c.name == "finite_values")
    assert finite_check.status == ValidationStatus.FAIL


def test_fail_infinity():
    contract = DataContract(
        name="no_infinity",
        required_modalities=[SensorModality.GNSS],
        min_frames=1,
        require_finite=True,
        max_nan_ratio=0.0,
    )
    frames = _make_frames(8)
    frames[3].observations[SensorModality.GNSS].data[0] = np.inf

    report = DataValidator(contract).validate_frames(frames)

    assert report.status == ValidationStatus.FAIL
    finite_check = next(c for c in report.checks if c.name == "finite_values")
    assert finite_check.status == ValidationStatus.FAIL
    assert finite_check.details["inf_count"] == 1


def test_fail_unsorted():
    contract = DataContract(
        name="sorted",
        required_modalities=[SensorModality.GNSS],
        min_frames=1,
        require_sorted_timestamps=True,
    )
    v = DataValidator(contract)
    report = v.validate_frames(_make_frames(6, shuffle=True))
    assert report.status == ValidationStatus.FAIL


def test_split_leakage():
    contract = DataContract(name="split", required_modalities=[])
    v = DataValidator(contract)
    report = v.validate_split_integrity(
        train_ids=["s1", "s2"],
        val_ids=["s2", "s3"],
        test_ids=["s4"],
    )
    assert report.status == ValidationStatus.FAIL
    assert report.summary["has_leakage"] is True


def test_split_clean():
    contract = DataContract(name="split", required_modalities=[])
    v = DataValidator(contract)
    report = v.validate_split_integrity(
        train_ids=["s1", "s2"],
        val_ids=["s3"],
        test_ids=["s4"],
    )
    assert report.status == ValidationStatus.PASS
