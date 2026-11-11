"""Consumer adapter for RoboQA GPS measurements wrapped by Vyra Core."""
from __future__ import annotations
from typing import Any
import numpy as np
from sensor_fusion_ml.data_engine.contracts import (
    CoordinateFrame, MeasurementUncertainty, Observation, SensorModality,
    Timestamp, TimeUnit, UncertaintyStatus,
)


class VyraEnvelopeError(ValueError):
    """Raised when a Vyra envelope lacks safe GNSS semantics."""


def roboqa_gps_envelope_to_observation(envelope: Any) -> Observation:
    """Convert a RoboQA GPS envelope into GNSS without coordinate transforms.

    Accepts only explicit local Cartesian x/y coordinates in metres.
    Geographic coordinates must be projected upstream by a validated transform.
    R is represented by typed Observation uncertainty; legacy metadata is retained.
    """
    for name in ("artifact_type", "payload", "temporal_reference",
                 "spatial_reference", "artifact_id", "producer"):
        if not hasattr(envelope, name):
            raise VyraEnvelopeError(f"Envelope is missing required field: {name}")
    if envelope.artifact_type != "gps_measurement":
        raise VyraEnvelopeError("Only artifact_type='gps_measurement' is supported")

    measurement = envelope.payload
    for name in ("t", "sensor", "z", "R"):
        if not hasattr(measurement, name):
            raise VyraEnvelopeError(f"Measurement payload is missing field: {name}")
    if not isinstance(measurement.sensor, str) or measurement.sensor.strip().lower() != "gps":
        raise VyraEnvelopeError("Measurement payload must declare sensor='gps'")

    z = np.asarray(measurement.z, dtype=np.float64)
    covariance = np.asarray(measurement.R, dtype=np.float64)
    if z.shape != (2,):
        raise VyraEnvelopeError(f"GPS z must have shape (2,), received {z.shape}")
    if covariance.shape != (2, 2):
        raise VyraEnvelopeError(f"GPS R must have shape (2, 2), received {covariance.shape}")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(covariance)):
        raise VyraEnvelopeError("GPS z and R must contain only finite values")
    if not np.allclose(covariance, covariance.T, atol=1e-8):
        raise VyraEnvelopeError("GPS R must be symmetric")
    if float(np.linalg.eigvalsh((covariance + covariance.T) / 2.0).min()) < -1e-8:
        raise VyraEnvelopeError("GPS R must be positive semidefinite")

    temporal = envelope.temporal_reference
    event_time = getattr(temporal, "event_time", None)
    if isinstance(event_time, bool) or not isinstance(event_time, (int, float, np.number)):
        raise VyraEnvelopeError("Numeric event_time is required for RoboQA GPS")
    if not np.isfinite(float(event_time)):
        raise VyraEnvelopeError("event_time must be finite")
    if float(event_time) != float(measurement.t):
        raise VyraEnvelopeError("Temporal event_time must match Measurement.t")
    if getattr(temporal, "event_time_unit", None) != "seconds":
        raise VyraEnvelopeError("RoboQA Measurement.t must be declared in seconds")
    time_reference = getattr(temporal, "time_reference", None)
    clock_domain = getattr(temporal, "clock_domain", None)
    if not time_reference or not clock_domain:
        raise VyraEnvelopeError("time_reference and clock_domain must be explicit")

    spatial = envelope.spatial_reference
    if getattr(spatial, "coordinate_frame", None) != "LOCAL_CARTESIAN":
        raise VyraEnvelopeError("Only LOCAL_CARTESIAN coordinates are accepted")
    if tuple(getattr(spatial, "axis_order", ()) or ()) != ("x", "y"):
        raise VyraEnvelopeError("Spatial axis_order must be exactly ('x', 'y')")
    if tuple(getattr(spatial, "units", ()) or ()) != ("m", "m"):
        raise VyraEnvelopeError("Spatial units must be exactly ('m', 'm')")
    if not getattr(spatial, "crs", None):
        raise VyraEnvelopeError("A local CRS/reference identifier is required")
    if not getattr(spatial, "coordinate_system", None):
        raise VyraEnvelopeError("coordinate_system must be explicit")
    if not getattr(spatial, "origin", None):
        raise VyraEnvelopeError("Spatial origin must be explicit")

    data = z.copy()
    data.setflags(write=False)
    covariance_copy = covariance.copy()
    covariance_copy.setflags(write=False)
    return Observation(
        modality=SensorModality.GNSS,
        timestamp=Timestamp(value=float(event_time), unit=TimeUnit.SECONDS,
                            source=str(clock_domain)),
        data=data,
        frame=CoordinateFrame.LOCAL_CARTESIAN,
        sensor_id=str(getattr(temporal, "source_id", None) or envelope.producer),
        quality=None,
        is_valid=True,
        is_estimate=False,
        uncertainty=MeasurementUncertainty(
            status=UncertaintyStatus.PROVIDED,
            covariance=covariance_copy,
            variable_order=("x", "y"),
            units=("m", "m"),
            frame=CoordinateFrame.LOCAL_CARTESIAN,
            provenance=f"vyra-artifact:{envelope.artifact_id}",
        ),
        metadata={
            "vyra_artifact_id": str(envelope.artifact_id),
            "producer": str(envelope.producer),
            "source_sensor": "gps",
            "quality_status": "not_provided_by_source",
            "measurement_covariance": covariance_copy.tolist(),
            "covariance_units": "m^2",
            "temporal_reference": {
                "event_time": float(event_time),
                "event_time_unit": "seconds",
                "time_reference": str(time_reference),
                "clock_domain": str(clock_domain),
                "source_id": getattr(temporal, "source_id", None),
                "sequence": getattr(temporal, "sequence", None),
                "synchronization": getattr(temporal, "synchronization", None),
                "synchronization_uncertainty": getattr(temporal, "synchronization_uncertainty", None),
            },
            "spatial_reference": {
                "coordinate_frame": spatial.coordinate_frame,
                "coordinate_system": spatial.coordinate_system,
                "crs": spatial.crs,
                "axis_order": tuple(spatial.axis_order),
                "units": tuple(spatial.units),
                "origin": spatial.origin,
                "transform_id": getattr(spatial, "transform_id", None),
                "transform_provenance": getattr(spatial, "transform_provenance", None),
                "spatial_uncertainty": getattr(spatial, "spatial_uncertainty", None),
            },
            "integration_policy": "preserve_position_and_covariance_no_transform",
        },
    )



class RoboQAGPSConsumerAdapter:
    """Vyra Core ConsumerAdapter for RoboQA GPS envelopes.

    The adapter is deliberately thin: all payload validation and Observation
    construction remain in ``roboqa_gps_envelope_to_observation``.
    """

    name = "roboqa-gps-consumer"
    version = "0.1.0"
    consumer_module_id = "vyra.sensor_fusion"

    def from_envelope(self, envelope: Any) -> Observation:
        """Convert one Core envelope into a typed GNSS Observation."""
        return roboqa_gps_envelope_to_observation(envelope)
