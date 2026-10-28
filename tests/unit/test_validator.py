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



def _check(report, name):
    return next(check for check in report.checks if check.name == name)


def test_fail_nonfinite_frame_timestamp():
    frames = _make_frames(5)
    frames[2].timestamp.value = np.nan
    report = DataValidator(DataContract(
        name="timestamp_finite",
        required_modalities=[SensorModality.GNSS],
    )).validate_frames(frames)
    assert report.status == ValidationStatus.BLOCKED
    assert _check(report, "timestamp_finiteness").status == ValidationStatus.FAIL
    assert _check(report, "timestamp_order").status == ValidationStatus.BLOCKED


def test_fail_nonfinite_observation_timestamp():
    frames = _make_frames(5)
    frames[2].observations[SensorModality.GNSS].timestamp.value = np.inf
    report = DataValidator(DataContract(
        name="observation_timestamp_finite",
        required_modalities=[SensorModality.GNSS],
    )).validate_frames(frames)
    assert report.status == ValidationStatus.FAIL
    assert _check(report, "timestamp_finiteness").status == ValidationStatus.FAIL


def test_observation_time_alignment_is_configurable():
    frames = _make_frames(5)
    frames[2].observations[SensorModality.GNSS].timestamp.value += 0.25
    contract = DataContract(
        name="alignment",
        required_modalities=[SensorModality.GNSS],
        max_observation_time_offset_s=0.1,
    )
    report = DataValidator(contract).validate_frames(frames)
    assert report.status == ValidationStatus.FAIL
    check = _check(report, "observation_time_alignment")
    assert check.status == ValidationStatus.FAIL
    assert check.details["max_offset_s"] == pytest.approx(0.25)


def test_nested_payload_checks_all_numeric_arrays():
    frames = _make_frames(5)
    frames[1].observations[SensorModality.GNSS].data = {
        "position": np.array([1.0, 2.0, 3.0]),
        "quality": {"residuals": np.array([0.0, np.inf])},
    }
    report = DataValidator(DataContract(
        name="nested_payload",
        required_modalities=[SensorModality.GNSS],
    )).validate_frames(frames)
    assert report.status == ValidationStatus.FAIL
    check = _check(report, "finite_values")
    assert check.details["inf_count"] == 1


def test_dimension_consistency_checks_fixed_shape_modalities():
    frames = _make_frames(5)
    frames[2].observations[SensorModality.GNSS].data = np.zeros(4)
    report = DataValidator(DataContract(
        name="gnss_shape",
        required_modalities=[SensorModality.GNSS],
    )).validate_frames(frames)
    check = _check(report, "dimension_consistency")
    assert check.status == ValidationStatus.PASS_WITH_WARNINGS
    assert check.details["inconsistent"][0]["modality"] == "gnss"


def test_dimension_consistency_allows_variable_lidar_point_counts():
    frames = _make_frames(5)
    frames[0].observations[SensorModality.LIDAR] = Observation(
        SensorModality.LIDAR, frames[0].timestamp, np.zeros((10, 3))
    )
    frames[1].observations[SensorModality.LIDAR] = Observation(
        SensorModality.LIDAR, frames[1].timestamp, np.zeros((25, 3))
    )
    report = DataValidator(DataContract(
        name="variable_lidar",
        required_modalities=[SensorModality.GNSS],
    )).validate_frames(frames)
    check = _check(report, "dimension_consistency")
    assert check.status == ValidationStatus.PASS


@pytest.mark.parametrize("ratio", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_nan_ratio_fails_contract_configuration(ratio):
    report = DataValidator(DataContract(
        name="invalid_ratio",
        required_modalities=[SensorModality.GNSS],
        max_nan_ratio=ratio,
    )).validate_frames(_make_frames(3))
    assert report.status == ValidationStatus.FAIL
    assert _check(report, "contract_configuration").status == ValidationStatus.FAIL
