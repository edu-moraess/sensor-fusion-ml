"""Fail-closed bridge from typed GNSS observations into the local ENU ESKF.

This module does not transform coordinates or synchronize clocks. It only
allows an update when those operations have already been declared and their
metadata satisfies this bridge's explicit contract.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from sensor_fusion_ml.data_engine.contracts import (
    CoordinateFrame,
    SensorModality,
    UncertaintyStatus,
)
from sensor_fusion_ml.navigation.vyra_eskf import (
    GNSS_2D_HORIZONTAL_MAPPING_ID,
)


class GNSSObservationBridgeError(ValueError):
    """Raised when an Observation is not safe for the ENU ESKF update."""


def apply_enu_gnss_observation(
    eskf: Any,
    observation: Any,
    *,
    state_reference_id: str,
    state_time_s: float,
    max_measurement_time_offset_s: float,
    max_synchronization_uncertainty_s: float,
) -> dict[str, Any]:
    """Apply a typed 2D GNSS Observation to VyraESKF under strict guards.

    Required semantics:
    - Observation frame and covariance frame are ENU.
    - Measurement axes are exactly (east, north), in metres.
    - Spatial CRS, origin, transform ID and transform provenance are explicit.
    - The ESKF state reference ID equals the canonical declared measurement
      reference formed from CRS and origin.
    - Source time is explicitly marked synchronized, its clock domain matches
      the Observation timestamp source, and its uncertainty is within the
      caller-supplied tolerance.
    - Measurement time is within a caller-supplied offset of the declared
      ESKF state time.
    - The 2x2 covariance is positive definite. PSD-but-singular covariance is
      preserved by the producer/consumer contract but is not regularized here.

    No coordinate transform, clock synchronization, covariance repair, or
    inferred metadata is performed.
    """
    if not callable(getattr(eskf, "update_gnss_horizontal", None)):
        raise GNSSObservationBridgeError(
            "eskf must expose update_gnss_horizontal"
        )
    if not isinstance(state_reference_id, str) or not state_reference_id.strip():
        raise GNSSObservationBridgeError("state_reference_id is required")
    if (
        isinstance(state_time_s, bool)
        or not isinstance(state_time_s, (int, float))
        or not math.isfinite(float(state_time_s))
    ):
        raise GNSSObservationBridgeError("state_time_s must be finite")
    if (
        isinstance(max_measurement_time_offset_s, bool)
        or not isinstance(max_measurement_time_offset_s, (int, float))
        or not math.isfinite(float(max_measurement_time_offset_s))
        or float(max_measurement_time_offset_s) < 0
    ):
        raise GNSSObservationBridgeError(
            "max_measurement_time_offset_s must be finite and non-negative"
        )
    if (
        isinstance(max_synchronization_uncertainty_s, bool)
        or not isinstance(max_synchronization_uncertainty_s, (int, float))
        or not math.isfinite(float(max_synchronization_uncertainty_s))
        or float(max_synchronization_uncertainty_s) < 0
    ):
        raise GNSSObservationBridgeError(
            "max_synchronization_uncertainty_s must be finite and non-negative"
        )

    if getattr(observation, "modality", None) != SensorModality.GNSS:
        raise GNSSObservationBridgeError("Observation modality must be GNSS")
    if getattr(observation, "frame", None) != CoordinateFrame.ENU:
        raise GNSSObservationBridgeError(
            "Only ENU observations may update the ENU ESKF; no implicit LOCAL_CARTESIAN conversion"
        )
    if getattr(observation, "is_valid", False) is not True:
        raise GNSSObservationBridgeError("Observation must be marked valid")
    if getattr(observation, "is_estimate", True) is not False:
        raise GNSSObservationBridgeError("GNSS input must be a direct measurement")

    timestamp = getattr(observation, "timestamp", None)
    if timestamp is None or not callable(getattr(timestamp, "to_seconds", None)):
        raise GNSSObservationBridgeError("Observation timestamp must support to_seconds()")
    timestamp_s = float(timestamp.to_seconds())
    if not math.isfinite(timestamp_s):
        raise GNSSObservationBridgeError("Observation timestamp must be finite")
    if abs(timestamp_s - float(state_time_s)) > float(max_measurement_time_offset_s):
        raise GNSSObservationBridgeError(
            "Measurement timestamp is outside the caller-supplied ESKF time tolerance"
        )

    uncertainty = getattr(observation, "uncertainty", None)
    if uncertainty is None or getattr(uncertainty, "status", None) != UncertaintyStatus.PROVIDED:
        raise GNSSObservationBridgeError("Typed PROVIDED measurement uncertainty is required")
    if getattr(uncertainty, "frame", None) != CoordinateFrame.ENU:
        raise GNSSObservationBridgeError("Covariance frame must be ENU")
    if tuple(getattr(uncertainty, "variable_order", ())) != ("east", "north"):
        raise GNSSObservationBridgeError(
            "Covariance variable_order must be exactly ('east', 'north')"
        )
    if tuple(getattr(uncertainty, "units", ())) != ("m", "m"):
        raise GNSSObservationBridgeError("Covariance units must be exactly ('m', 'm')")

    position = np.asarray(getattr(observation, "data", None), dtype=np.float64)
    covariance = np.asarray(getattr(uncertainty, "covariance", None), dtype=np.float64)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise GNSSObservationBridgeError("ENU position must be a finite shape-(2,) vector")
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise GNSSObservationBridgeError("ENU covariance must be a finite 2x2 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-9):
        raise GNSSObservationBridgeError("ENU covariance must be symmetric")
    if float(np.linalg.eigvalsh((covariance + covariance.T) / 2.0).min()) <= 0.0:
        raise GNSSObservationBridgeError(
            "ESKF requires positive-definite covariance; singular PSD covariance is not regularized"
        )

    metadata = getattr(observation, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise GNSSObservationBridgeError("Observation metadata mapping is required")

    spatial = metadata.get("spatial_reference")
    if not isinstance(spatial, Mapping):
        raise GNSSObservationBridgeError("Explicit spatial_reference metadata is required")
    if spatial.get("coordinate_frame") != "ENU":
        raise GNSSObservationBridgeError("Spatial metadata coordinate_frame must be ENU")
    if tuple(spatial.get("axis_order") or ()) != ("east", "north"):
        raise GNSSObservationBridgeError(
            "Spatial metadata axis_order must be exactly ('east', 'north')"
        )
    if tuple(spatial.get("units") or ()) != ("m", "m"):
        raise GNSSObservationBridgeError("Spatial metadata units must be exactly ('m', 'm')")

    crs = spatial.get("crs")
    origin = spatial.get("origin")
    transform_id = spatial.get("transform_id")
    transform_provenance = spatial.get("transform_provenance")
    for name, value in (
        ("crs", crs),
        ("origin", origin),
        ("transform_id", transform_id),
        ("transform_provenance", transform_provenance),
    ):
        if not isinstance(value, str) or not value.strip():
            raise GNSSObservationBridgeError(
                f"spatial_reference.{name} must be explicitly declared"
            )

    measurement_reference_id = f"{crs}::{origin}"
    if measurement_reference_id != state_reference_id:
        raise GNSSObservationBridgeError(
            "Measurement spatial reference does not match the ESKF state reference"
        )

    temporal = metadata.get("temporal_reference")
    if not isinstance(temporal, Mapping):
        raise GNSSObservationBridgeError("Explicit temporal_reference metadata is required")
    clock_domain = temporal.get("clock_domain")
    timestamp_source = getattr(timestamp, "source", None)
    if not isinstance(clock_domain, str) or not clock_domain.strip():
        raise GNSSObservationBridgeError("temporal_reference.clock_domain is required")
    if timestamp_source != clock_domain:
        raise GNSSObservationBridgeError(
            "Observation timestamp source must match temporal clock_domain"
        )
    if temporal.get("synchronization") != "synchronized":
        raise GNSSObservationBridgeError(
            "GNSS source must be explicitly synchronized with the ESKF time base"
        )
    synchronization_uncertainty = temporal.get("synchronization_uncertainty")
    if (
        isinstance(synchronization_uncertainty, bool)
        or not isinstance(synchronization_uncertainty, (int, float))
        or not math.isfinite(float(synchronization_uncertainty))
        or float(synchronization_uncertainty) < 0
    ):
        raise GNSSObservationBridgeError(
            "A finite non-negative synchronization_uncertainty is required"
        )
    if float(synchronization_uncertainty) > float(max_synchronization_uncertainty_s):
        raise GNSSObservationBridgeError(
            "Synchronization uncertainty exceeds the caller-supplied tolerance"
        )

    report = eskf.update_gnss_horizontal(
        position,
        covariance,
        mapping_id=GNSS_2D_HORIZONTAL_MAPPING_ID,
        measurement_reference_id=measurement_reference_id,
        state_reference_id=state_reference_id,
    )
    return {
        "measurement_timestamp_s": timestamp_s,
        "eskf_state_time_s": float(state_time_s),
        "measurement_time_offset_s": abs(timestamp_s - float(state_time_s)),
        "measurement_reference_id": measurement_reference_id,
        "synchronization_uncertainty_s": float(synchronization_uncertainty),
        "mapping_id": GNSS_2D_HORIZONTAL_MAPPING_ID,
        "eskf_report": report,
    }
