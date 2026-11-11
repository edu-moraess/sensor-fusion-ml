"""Consumer for native MPE Detection artifacts carried by Vyra Core."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import math
from typing import Any

from .vyra_core import VyraEnvelopeError

from sensor_fusion_ml.data_engine.contracts import (
    CoordinateFrame,
    Observation,
    SensorModality,
    TimeUnit,
    Timestamp,
)


_EVIDENCE_TO_IS_ESTIMATE = {
    "MEASURED": False,
    "INFERRED": True,
    "DERIVED": True,
    "PREDICTED": True,
    "HYPOTHESIS": True,
}


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VyraEnvelopeError(f"{field} must be a non-empty string")
    return value


def _provenance_payload(provenance: Any) -> dict[str, Any]:
    names = (
        "producer", "source_artifact_ids", "producer_version", "adapter",
        "adapter_version", "run_id", "config_digest", "content_digest",
        "evidence_kind", "determinism", "seed",
    )
    result: dict[str, Any] = {}
    for name in names:
        value = getattr(provenance, name, None)
        if isinstance(value, tuple):
            value = list(value)
        if hasattr(value, "value"):
            value = value.value
        result[name] = value
    return result


def mpe_detection_envelope_to_observation(envelope: Any) -> Observation:
    """Map one native MPE Detection envelope into a PIXEL Observation.

    MPE and Vyra Core are optional integration dependencies and are imported
    only when this adapter is called. No pixel-to-metric transform, clipping,
    calibration, tracking, or cross-sensor synchronization is performed.
    """
    try:
        from mpe.domain.contracts import CONTRACTS_VERSION, Detection
        from mpe.domain.geometry import (
            IMAGE_BBOX_XYWH_V1,
            GeometryProfileError,
            validate_image_bbox_xywh,
        )
        from vyra_core.contracts import ArtifactEnvelope, SpatialReference
    except ImportError as exc:
        raise VyraEnvelopeError(
            "MPE and Vyra Core packages must be installed to use this adapter"
        ) from exc

    if not isinstance(envelope, ArtifactEnvelope):
        raise VyraEnvelopeError("envelope must be a Vyra Core ArtifactEnvelope")
    if envelope.artifact_type != "detection" or envelope.schema_name != "mpe.detection":
        raise VyraEnvelopeError("Only MPE artifact_type='detection' is supported")
    if envelope.schema_version != CONTRACTS_VERSION:
        raise VyraEnvelopeError(
            f"Unsupported MPE schema version: {envelope.schema_version}"
        )
    if not isinstance(envelope.payload, Detection):
        raise VyraEnvelopeError("Envelope payload must be a native MPE Detection")
    detection = envelope.payload
    if not isinstance(envelope.spatial_reference, SpatialReference):
        raise VyraEnvelopeError("An explicit Vyra PIXEL SpatialReference is required")
    if envelope.temporal_reference is None:
        raise VyraEnvelopeError("An explicit temporal reference is required")

    detection_id = _required_text(detection.detection_id, "detection_id")
    class_label = _required_text(detection.class_label, "class_label")
    stamp = detection.stamp
    source_id = _required_text(getattr(stamp, "source_id", None), "stamp.source_id")
    clock_domain = _required_text(getattr(stamp, "clock_domain", None), "stamp.clock_domain")
    sequence = getattr(stamp, "sequence", None)
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise VyraEnvelopeError("stamp.sequence must be a non-negative integer")
    capture_time = getattr(stamp, "capture_timestamp", None)
    if not isinstance(capture_time, datetime) or capture_time.tzinfo is None or capture_time.utcoffset() is None:
        raise VyraEnvelopeError("Detection capture_timestamp must be timezone-aware")

    temporal = envelope.temporal_reference
    if temporal.event_time != capture_time:
        raise VyraEnvelopeError("Envelope event_time must match Detection capture_timestamp")
    if temporal.source_id != source_id:
        raise VyraEnvelopeError("Envelope source_id must match Detection Stamp")
    if temporal.sequence != sequence:
        raise VyraEnvelopeError("Envelope sequence must match Detection Stamp")
    if temporal.clock_domain != clock_domain:
        raise VyraEnvelopeError("Envelope clock_domain must match Detection Stamp")

    spatial = envelope.spatial_reference
    if spatial.coordinate_frame != "PIXEL":
        raise VyraEnvelopeError("Spatial coordinate_frame must be PIXEL")
    if spatial.coordinate_system != "IMAGE_2D":
        raise VyraEnvelopeError("Spatial coordinate_system must be IMAGE_2D")
    if spatial.crs is not None:
        raise VyraEnvelopeError("PIXEL image coordinates must not invent a CRS")
    if tuple(spatial.axis_order or ()) != ("u", "v"):
        raise VyraEnvelopeError("PIXEL axis_order must be exactly ('u', 'v')")
    if tuple(spatial.units or ()) != ("px", "px"):
        raise VyraEnvelopeError("PIXEL units must be exactly ('px', 'px')")
    if spatial.spatial_uncertainty is not None:
        raise VyraEnvelopeError("Scalar spatial_uncertainty is not defined for this profile")

    try:
        geometry = validate_image_bbox_xywh(detection.geometry)
    except GeometryProfileError as exc:
        raise VyraEnvelopeError(f"Invalid MPE image bbox geometry: {exc}") from exc
    image = geometry["image"]
    image_reference = _required_text(image["reference"], "geometry.image.reference")
    image_source_id = _required_text(image["source_id"], "geometry.image.source_id")
    image_sequence = image["sequence"]
    if image_source_id != source_id:
        raise VyraEnvelopeError(
            "Geometry image source_id must match Detection Stamp source_id"
        )
    if image_sequence != sequence:
        raise VyraEnvelopeError(
            "Geometry image sequence must match Detection Stamp sequence"
        )
    expected_origin = f"top-left:{image_reference}"
    if spatial.origin != expected_origin:
        raise VyraEnvelopeError(
            "PIXEL SpatialReference origin must identify the geometry image reference"
        )

    preprocessing = geometry["preprocessing"]
    if preprocessing["applied"]:
        if spatial.transform_id != preprocessing["transform_id"]:
            raise VyraEnvelopeError("Spatial transform_id must match geometry preprocessing")
        if spatial.transform_provenance != preprocessing["transform_provenance"]:
            raise VyraEnvelopeError(
                "Spatial transform_provenance must match geometry preprocessing"
            )
    elif spatial.transform_id is not None or spatial.transform_provenance is not None:
        raise VyraEnvelopeError(
            "Spatial transforms must be absent when geometry is already in original_image space"
        )

    evidence = getattr(detection.evidence_kind, "value", detection.evidence_kind)
    evidence = _required_text(evidence, "evidence_kind").upper()
    if evidence == "EXTERNAL":
        raise VyraEnvelopeError(
            "EvidenceKind.EXTERNAL requires an explicit source-specific policy"
        )
    if evidence not in _EVIDENCE_TO_IS_ESTIMATE:
        raise VyraEnvelopeError(f"Unsupported MPE evidence_kind: {evidence}")

    try:
        timestamp_seconds = capture_time.timestamp()
    except (OverflowError, OSError, ValueError) as exc:
        raise VyraEnvelopeError("Capture timestamp cannot be represented as Unix seconds") from exc
    if not math.isfinite(timestamp_seconds):
        raise VyraEnvelopeError("Capture timestamp must convert to finite Unix seconds")

    extensions = envelope.extensions or {}
    mpe_extensions = extensions.get("mpe", {}) if isinstance(extensions, Mapping) else {}
    if not isinstance(mpe_extensions, Mapping):
        raise VyraEnvelopeError("Envelope extensions['mpe'] must be a mapping")
    if mpe_extensions.get("contracts_version") != CONTRACTS_VERSION:
        raise VyraEnvelopeError(
            "MPE extension contracts_version must match the envelope schema"
        )

    provenance = _provenance_payload(envelope.provenance)
    original_timestamp = capture_time.isoformat()
    metadata = {
        "vyra_artifact_id": envelope.artifact_id,
        "mpe_detection_id": detection_id,
        "producer": envelope.producer,
        "source_repository": mpe_extensions.get("source_repository"),
        "source_commit": mpe_extensions.get("source_commit"),
        "model_ref": detection.model_ref,
        "detector_score": detection.score,
        "mpe_quality": detection.quality,
        "evidence_kind": evidence,
        "geometry_profile": IMAGE_BBOX_XYWH_V1,
        "image_reference": image_reference,
        "image_source_id": image_source_id,
        "image_sequence": image_sequence,
        "image_width": image["width"],
        "image_height": image["height"],
        "capture_timestamp_original": original_timestamp,
        "clock_domain": clock_domain,
        "source_sequence": sequence,
        "source_id": source_id,
        "provenance": provenance,
        "mpe_extensions": dict(mpe_extensions),
        "spatial_reference": {
            "coordinate_frame": spatial.coordinate_frame,
            "coordinate_system": spatial.coordinate_system,
            "crs": spatial.crs,
            "axis_order": tuple(spatial.axis_order),
            "units": tuple(spatial.units),
            "origin": spatial.origin,
            "transform_id": spatial.transform_id,
            "transform_provenance": spatial.transform_provenance,
        },
        "integration_policy": "preserve_original_image_pixels_no_projection",
    }

    return Observation(
        modality=SensorModality.DETECTION,
        timestamp=Timestamp(
            value=float(timestamp_seconds),
            unit=TimeUnit.SECONDS,
            source=clock_domain,
        ),
        data={
            "geometry_profile": IMAGE_BBOX_XYWH_V1,
            "geometry": geometry,
            "class_label": class_label,
        },
        frame=CoordinateFrame.PIXEL,
        sensor_id=source_id,
        quality=None,
        is_valid=True,
        is_estimate=_EVIDENCE_TO_IS_ESTIMATE[evidence],
        metadata=metadata,
    )
