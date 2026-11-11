import numpy as np
import pytest

from sensor_fusion_ml.data_engine.contracts import (
    CoordinateFrame,
    MeasurementUncertainty,
    Observation,
    SensorModality,
    TimeUnit,
    Timestamp,
    UncertaintyStatus,
)
from sensor_fusion_ml.navigation import (
    GNSSObservationBridgeError,
    VyraESKF,
    apply_enu_gnss_observation,
)


CRS = "LOCAL_ENU:WGS84:origin-A"
ORIGIN = "origin-A"
REFERENCE_ID = f"{CRS}::{ORIGIN}"


def make_observation(
    *,
    frame=CoordinateFrame.ENU,
    covariance=None,
    synchronization="synchronized",
    synchronization_uncertainty=0.001,
    axis_order=("east", "north"),
    transform_id="explicit-transform-v1",
):
    if covariance is None:
        covariance = np.array([[0.25, 0.02], [0.02, 0.36]])
    uncertainty = MeasurementUncertainty(
        status=UncertaintyStatus.PROVIDED,
        covariance=covariance,
        variable_order=("east", "north"),
        units=("m", "m"),
        frame=frame,
        provenance="test-artifact:gnss-01",
    )
    return Observation(
        modality=SensorModality.GNSS,
        timestamp=Timestamp(1.25, TimeUnit.SECONDS, source="roboqa-test-clock"),
        data=np.array([12.0, 34.0]),
        frame=frame,
        sensor_id="gps-01",
        is_valid=True,
        is_estimate=False,
        uncertainty=uncertainty,
        metadata={
            "spatial_reference": {
                "coordinate_frame": "ENU" if frame == CoordinateFrame.ENU else "LOCAL_CARTESIAN",
                "coordinate_system": "LOCAL_TANGENT_ENU",
                "crs": CRS,
                "axis_order": axis_order,
                "units": ("m", "m"),
                "origin": ORIGIN,
                "transform_id": transform_id,
                "transform_provenance": "test fixture; not a production transform",
            },
            "temporal_reference": {
                "synchronization": synchronization,
                "synchronization_uncertainty": synchronization_uncertainty,
            },
        },
    )


def test_valid_explicit_enu_observation_updates_eskf():
    observation = make_observation()
    eskf = VyraESKF(P=np.eye(15) * 2.0)

    result = apply_enu_gnss_observation(
        eskf,
        observation,
        state_reference_id=REFERENCE_ID,
        max_synchronization_uncertainty_s=0.01,
    )

    assert result["mapping_id"] == "gnss_2d_horizontal_v1"
    assert result["measurement_reference_id"] == REFERENCE_ID
    assert result["eskf_report"]["vertical_measurement_status"] == "NOT_OBSERVED"
    assert eskf.history[-1]["event"] == "gnss_horizontal"


def test_local_cartesian_roboqa_observation_is_not_implicitly_treated_as_enu():
    observation = make_observation(frame=CoordinateFrame.LOCAL_CARTESIAN)
    eskf = VyraESKF()

    with pytest.raises(GNSSObservationBridgeError, match="Only ENU observations"):
        apply_enu_gnss_observation(
            eskf,
            observation,
            state_reference_id=REFERENCE_ID,
            max_synchronization_uncertainty_s=0.01,
        )

    assert eskf.history == []


def test_unsynchronized_roboqa_time_is_rejected():
    observation = make_observation(synchronization="not_synchronized")
    with pytest.raises(GNSSObservationBridgeError, match="explicitly synchronized"):
        apply_enu_gnss_observation(
            VyraESKF(),
            observation,
            state_reference_id=REFERENCE_ID,
            max_synchronization_uncertainty_s=0.01,
        )


def test_synchronization_uncertainty_must_be_within_caller_tolerance():
    observation = make_observation(synchronization_uncertainty=0.02)
    with pytest.raises(GNSSObservationBridgeError, match="exceeds"):
        apply_enu_gnss_observation(
            VyraESKF(),
            observation,
            state_reference_id=REFERENCE_ID,
            max_synchronization_uncertainty_s=0.01,
        )


def test_measurement_and_state_spatial_references_must_match():
    observation = make_observation()
    with pytest.raises(GNSSObservationBridgeError, match="does not match"):
        apply_enu_gnss_observation(
            VyraESKF(),
            observation,
            state_reference_id="different-crs::different-origin",
            max_synchronization_uncertainty_s=0.01,
        )


def test_spatial_transform_identity_and_provenance_are_required():
    observation = make_observation(transform_id=None)
    with pytest.raises(GNSSObservationBridgeError, match="transform_id"):
        apply_enu_gnss_observation(
            VyraESKF(),
            observation,
            state_reference_id=REFERENCE_ID,
            max_synchronization_uncertainty_s=0.01,
        )


def test_singular_psd_covariance_is_not_regularized_for_eskf():
    observation = make_observation(covariance=np.diag([0.25, 0.0]))
    with pytest.raises(GNSSObservationBridgeError, match="positive-definite"):
        apply_enu_gnss_observation(
            VyraESKF(),
            observation,
            state_reference_id=REFERENCE_ID,
            max_synchronization_uncertainty_s=0.01,
        )


def test_covariance_axes_must_be_east_north():
    observation = make_observation(axis_order=("north", "east"))
    with pytest.raises(GNSSObservationBridgeError, match="axis_order"):
        apply_enu_gnss_observation(
            VyraESKF(),
            observation,
            state_reference_id=REFERENCE_ID,
            max_synchronization_uncertainty_s=0.01,
        )
