from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

mpe_contracts = pytest.importorskip("mpe.domain.contracts")
mpe_core_kinds = pytest.importorskip("mpe.core.kinds")
mpe_core_origin = pytest.importorskip("mpe.core.origin")
mpe_core_provenance = pytest.importorskip("mpe.core.provenance")
mpe_core_stamp = pytest.importorskip("mpe.core.stamp")
mpe_geometry = pytest.importorskip("mpe.domain.geometry")
vyra_contracts = pytest.importorskip("vyra_core.contracts")
vyra_adapter = pytest.importorskip("vyra_core.adapters.mpe")

from sensor_fusion_ml.data_engine.contracts import CoordinateFrame, SensorModality, TimeUnit
from sensor_fusion_ml.integrations.mpe_detection import (
    VyraEnvelopeError,
    mpe_detection_envelope_to_observation,
)

UTC = timezone.utc
CAPTURE_TIME = datetime(2026, 11, 11, 10, 0, 0, tzinfo=UTC)
IMAGE_REF = "camera-front/frame-000123"


def make_detection(evidence_kind=None, geometry=None):
    kind = evidence_kind or mpe_core_kinds.EvidenceKind.INFERRED
    provenance = mpe_core_provenance.Provenance(
        origin=mpe_core_origin.Origin(
            mpe_core_origin.OriginKind.SYNTHETIC, "camera-front"
        ),
        timestamp=CAPTURE_TIME,
        parent_artifact_ids=("frame-artifact-123",),
        pipeline_version="0.3.0",
        run_id="run-a3-fixture",
    )
    return mpe_contracts.Detection(
        detection_id="det-000123",
        stamp=mpe_core_stamp.Stamp(
            source_id="camera-front",
            sequence=123,
            capture_timestamp=CAPTURE_TIME,
            clock_domain="camera-monotonic",
        ),
        class_label="vehicle",
        geometry=geometry or {
            "type": "bbox",
            "profile": "mpe.image_bbox_xywh.v1",
            "coordinates": {"x": 120.0, "y": 80.0, "w": 64.0, "h": 96.0},
            "image": {
                "width": 1280,
                "height": 720,
                "reference": IMAGE_REF,
                "coordinate_space": "original_image",
            },
            "preprocessing": {
                "applied": False,
                "transform_id": None,
                "transform_provenance": None,
            },
        },
        score=0.93,
        provenance=provenance,
        evidence_kind=kind,
        model_ref="detector-v1",
        quality="valid",
    )


def make_envelope(detection=None, spatial=None):
    detection = detection or make_detection()
    spatial = spatial or vyra_contracts.SpatialReference(
        coordinate_frame="PIXEL",
        coordinate_system="IMAGE_2D",
        crs=None,
        axis_order=("u", "v"),
        units=("px", "px"),
        origin=f"top-left:{IMAGE_REF}",
    )
    producer = vyra_adapter.MPEProducerAdapter(
        spatial_reference=spatial,
        source_repository="edu-moraess/vyra-mpe-intell",
        source_commit="fixture-commit",
    )
    return producer.to_envelope(detection)


def test_native_mpe_detection_maps_to_pixel_observation_without_projection():
    detection = make_detection()
    envelope = make_envelope(detection)
    observation = mpe_detection_envelope_to_observation(envelope)

    assert observation.modality is SensorModality.DETECTION
    assert observation.frame is CoordinateFrame.PIXEL
    assert observation.sensor_id == "camera-front"
    assert observation.timestamp.unit is TimeUnit.SECONDS
    assert observation.timestamp.source == "camera-monotonic"
    assert observation.timestamp.value == pytest.approx(CAPTURE_TIME.timestamp())
    assert observation.quality is None
    assert observation.is_estimate is True
    assert observation.data["geometry"] is detection.geometry
    assert observation.data["class_label"] == "vehicle"
    assert observation.metadata["vyra_artifact_id"] == envelope.artifact_id
    assert observation.metadata["mpe_detection_id"] == detection.detection_id
    assert observation.metadata["detector_score"] == 0.93
    assert observation.metadata["mpe_quality"] == "valid"
    assert observation.metadata["evidence_kind"] == "INFERRED"
    assert observation.metadata["source_repository"] == "edu-moraess/vyra-mpe-intell"
    assert observation.metadata["source_commit"] == "fixture-commit"
    assert observation.metadata["image_reference"] == IMAGE_REF
    assert observation.metadata["image_width"] == 1280
    assert observation.metadata["image_height"] == 720
    assert observation.metadata["clock_domain"] == "camera-monotonic"
    assert observation.metadata["source_sequence"] == 123
    assert observation.metadata["capture_timestamp_original"] == CAPTURE_TIME.isoformat()
    assert observation.metadata["provenance"]["source_artifact_ids"] == ["frame-artifact-123"]
    assert "measurement_covariance" not in observation.metadata


@pytest.mark.parametrize(
    "kind, expected",
    [
        (mpe_core_kinds.EvidenceKind.MEASURED, False),
        (mpe_core_kinds.EvidenceKind.INFERRED, True),
        (mpe_core_kinds.EvidenceKind.DERIVED, True),
        (mpe_core_kinds.EvidenceKind.PREDICTED, True),
        (mpe_core_kinds.EvidenceKind.HYPOTHESIS, True),
    ],
)
def test_evidence_kind_mapping_is_explicit(kind, expected):
    observation = mpe_detection_envelope_to_observation(make_envelope(make_detection(kind)))
    assert observation.is_estimate is expected
    assert observation.metadata["evidence_kind"] == kind.value


def test_external_evidence_is_rejected_without_source_policy():
    envelope = make_envelope(make_detection(mpe_core_kinds.EvidenceKind.EXTERNAL))
    with pytest.raises(VyraEnvelopeError, match="source-specific policy"):
        mpe_detection_envelope_to_observation(envelope)


def test_rejects_invalid_geometry_profile_and_bounds():
    detection = make_detection()
    bad_geometry = dict(detection.geometry)
    bad_geometry["coordinates"] = {"x": 1270, "y": 80, "w": 64, "h": 96}
    envelope = make_envelope(replace(detection, geometry=bad_geometry))
    with pytest.raises(VyraEnvelopeError, match="geometry"):
        mpe_detection_envelope_to_observation(envelope)


def test_rejects_missing_geometry_profile():
    detection = make_detection()
    bad_geometry = dict(detection.geometry)
    bad_geometry.pop("profile")
    envelope = make_envelope(replace(detection, geometry=bad_geometry))
    with pytest.raises(VyraEnvelopeError, match="profile"):
        mpe_detection_envelope_to_observation(envelope)


def test_rejects_spatial_reference_image_mismatch():
    spatial = vyra_contracts.SpatialReference(
        coordinate_frame="PIXEL",
        coordinate_system="IMAGE_2D",
        axis_order=("u", "v"),
        units=("px", "px"),
        origin="top-left:another-image/frame-1",
    )
    envelope = make_envelope(spatial=spatial)
    with pytest.raises(VyraEnvelopeError, match="origin"):
        mpe_detection_envelope_to_observation(envelope)


def test_rejects_wrong_pixel_axes_and_units():
    spatial = vyra_contracts.SpatialReference(
        coordinate_frame="PIXEL",
        coordinate_system="IMAGE_2D",
        axis_order=("v", "u"),
        units=("px", "px"),
        origin=f"top-left:{IMAGE_REF}",
    )
    with pytest.raises(VyraEnvelopeError, match="axis_order"):
        mpe_detection_envelope_to_observation(make_envelope(spatial=spatial))


def test_rejects_temporal_metadata_mismatch():
    envelope = make_envelope()
    temporal = replace(envelope.temporal_reference, sequence=124)
    with pytest.raises(VyraEnvelopeError, match="sequence"):
        mpe_detection_envelope_to_observation(replace(envelope, temporal_reference=temporal))


def test_rejects_naive_capture_timestamp():
    detection = make_detection()
    naive_stamp = replace(detection.stamp, capture_timestamp=datetime(2026, 11, 11, 10))
    envelope = make_envelope(replace(detection, stamp=naive_stamp))
    with pytest.raises(VyraEnvelopeError, match="timezone-aware"):
        mpe_detection_envelope_to_observation(envelope)


def test_preprocessing_requires_matching_transform_metadata():
    detection = make_detection()
    geometry = dict(detection.geometry)
    geometry["preprocessing"] = {
        "applied": True,
        "transform_id": "letterbox-inverse-v1",
        "transform_provenance": "model-to-original-image mapping",
    }
    spatial = vyra_contracts.SpatialReference(
        coordinate_frame="PIXEL",
        coordinate_system="IMAGE_2D",
        axis_order=("u", "v"),
        units=("px", "px"),
        origin=f"top-left:{IMAGE_REF}",
        transform_id="wrong-transform",
        transform_provenance="model-to-original-image mapping",
    )
    envelope = make_envelope(replace(detection, geometry=geometry), spatial)
    with pytest.raises(VyraEnvelopeError, match="transform_id"):
        mpe_detection_envelope_to_observation(envelope)


def test_wrong_artifact_type_is_rejected():
    envelope = make_envelope()
    with pytest.raises(VyraEnvelopeError, match="artifact_type='detection'"):
        mpe_detection_envelope_to_observation(replace(envelope, artifact_type="track"))
