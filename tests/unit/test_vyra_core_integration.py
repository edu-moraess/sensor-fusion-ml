import json
import numpy as np
import pytest
from types import SimpleNamespace
from sensor_fusion_ml.integrations.vyra_core import VyraEnvelopeError, roboqa_gps_envelope_to_observation


def make_envelope():
    measurement = SimpleNamespace(t=1.25, sensor="gps", z=np.array([12.0, 34.0]),
                                  R=np.array([[4.0, 0.5], [0.5, 9.0]]))
    temporal = SimpleNamespace(event_time=1.25, event_time_unit="seconds",
        time_reference="sequence_elapsed", clock_domain="roboqa_sequence",
        source_id="oxts-0001", sequence=12, synchronization="not_synchronized",
        synchronization_uncertainty=None)
    spatial = SimpleNamespace(coordinate_frame="LOCAL_CARTESIAN",
        coordinate_system="local_tangent_plane_equirectangular",
        crs="LOCAL:WGS84:first-oxts-sample", axis_order=("x", "y"),
        units=("m", "m"), origin="first OXTS latitude/longitude sample",
        transform_id=None, transform_provenance="RoboQA latlon_to_local",
        spatial_uncertainty=None)
    return SimpleNamespace(artifact_type="gps_measurement", payload=measurement,
        temporal_reference=temporal, spatial_reference=spatial,
        artifact_id="artifact-1", producer="roboqa-lab")


def test_converts_local_metric_gps_and_preserves_covariance():
    envelope = make_envelope()
    observation = roboqa_gps_envelope_to_observation(envelope)
    assert observation.modality.value == "gnss"
    assert observation.frame.value == "local_cartesian"
    assert observation.timestamp.value == 1.25
    assert observation.quality is None
    assert observation.metadata["quality_status"] == "not_provided_by_source"
    np.testing.assert_array_equal(observation.data, [12.0, 34.0])
    assert not observation.data.flags.writeable
    assert observation.metadata["measurement_covariance"] == envelope.payload.R.tolist()
    assert observation.uncertainty.status.value == "provided"
    assert observation.uncertainty.variable_order == ("x", "y")
    assert observation.uncertainty.units == ("m", "m")
    assert observation.uncertainty.frame.value == "local_cartesian"
    np.testing.assert_array_equal(observation.uncertainty.covariance, envelope.payload.R)
    json.dumps(observation.to_dict())


def test_rejects_geographic_coordinates():
    envelope = make_envelope()
    envelope.spatial_reference.coordinate_frame = "GEOGRAPHIC"
    with pytest.raises(VyraEnvelopeError, match="LOCAL_CARTESIAN"):
        roboqa_gps_envelope_to_observation(envelope)


@pytest.mark.parametrize("field,value", [
    ("axis_order", ("y", "x")), ("units", ("degrees", "degrees")),
    ("crs", None), ("origin", None),
])
def test_rejects_incomplete_spatial_reference(field, value):
    envelope = make_envelope()
    setattr(envelope.spatial_reference, field, value)
    with pytest.raises(VyraEnvelopeError):
        roboqa_gps_envelope_to_observation(envelope)


def test_rejects_missing_clock_semantics():
    envelope = make_envelope()
    envelope.temporal_reference.clock_domain = None
    with pytest.raises(VyraEnvelopeError, match="clock_domain"):
        roboqa_gps_envelope_to_observation(envelope)


def test_rejects_time_mismatch():
    envelope = make_envelope()
    envelope.temporal_reference.event_time = 2.0
    with pytest.raises(VyraEnvelopeError, match="match Measurement.t"):
        roboqa_gps_envelope_to_observation(envelope)


def test_rejects_invalid_covariance():
    envelope = make_envelope()
    envelope.payload.R = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(VyraEnvelopeError, match="positive semidefinite"):
        roboqa_gps_envelope_to_observation(envelope)


def test_rejects_wrong_payload_type():
    envelope = make_envelope()
    envelope.artifact_type = "track"
    with pytest.raises(VyraEnvelopeError, match="gps_measurement"):
        roboqa_gps_envelope_to_observation(envelope)
