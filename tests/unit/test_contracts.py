"""Unit tests for data contracts."""

import json
import numpy as np
import pytest

from sensor_fusion_ml.data_engine.contracts import (
    DataContract,
    Frame,
    Observation,
    SensorModality,
    Timestamp,
    TimeUnit,
    Pose3D,
    CoordinateFrame,
)


def test_timestamp_conversion():
    t = Timestamp(value=1_000_000_000, unit=TimeUnit.NANOSECONDS)
    assert abs(t.to_seconds() - 1.0) < 1e-9
    t2 = Timestamp(value=1500, unit=TimeUnit.MILLISECONDS)
    assert abs(t2.to_seconds() - 1.5) < 1e-9


def test_pose3d_validation():
    p = Pose3D(position=np.zeros(3), orientation=np.array([1.0, 0, 0, 0]))
    assert p.position.shape == (3,)
    with pytest.raises(ValueError):
        Pose3D(position=np.zeros(2), orientation=np.array([1.0, 0, 0, 0]))


def test_frame_available_modalities():
    ts = Timestamp(0.0)
    obs = {
        SensorModality.GNSS: Observation(SensorModality.GNSS, ts, np.zeros(3), is_valid=True),
        SensorModality.IMU: Observation(SensorModality.IMU, ts, np.zeros(6), is_valid=False),
    }
    f = Frame("f0", ts, obs)
    assert SensorModality.GNSS in f.available_modalities()
    assert SensorModality.IMU not in f.available_modalities()
    assert f.has(SensorModality.GNSS)
    assert not f.has(SensorModality.IMU)


def test_data_contract_to_dict():
    c = DataContract(
        name="test",
        required_modalities=[SensorModality.GNSS],
        license="MIT",
    )
    d = c.to_dict()
    assert d["name"] == "test"
    assert "gnss" in d["required_modalities"]
    assert d["license"] == "MIT"


def _make_observation(quality_marker=...):
    kwargs = {} if quality_marker is ... else {"quality": quality_marker}
    return Observation(
        modality=SensorModality.DETECTION,
        timestamp=Timestamp(1.0),
        data={"bbox": [1, 2, 3, 4]},
        **kwargs,
    )


def test_observation_quality_defaults_to_none_and_serializes():
    observation = _make_observation()
    assert observation.quality is None
    assert observation.to_dict()["quality"] is None
    json.dumps(observation.to_dict())


@pytest.mark.parametrize("quality", [0.0, 0.25, 1.0, np.float32(0.5)])
def test_observation_accepts_finite_quality_in_range(quality):
    observation = _make_observation(quality)
    assert observation.quality == pytest.approx(float(quality))
    assert isinstance(observation.quality, float)


@pytest.mark.parametrize(
    "quality",
    [True, False, "0.5", float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
)
def test_observation_rejects_invalid_quality(quality):
    with pytest.raises(ValueError, match="quality"):
        _make_observation(quality)
