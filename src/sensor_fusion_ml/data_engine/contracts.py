"""
Data contracts and sensor modality definitions.

All numerical values that are estimates (not direct measurements) must be
explicitly flagged. Coordinates from different sensors must never be mixed
without documented, validated transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class SensorModality(str, Enum):
    RGB = "rgb"
    LIDAR = "lidar"
    RADAR = "radar"
    GNSS = "gnss"
    IMU = "imu"
    ODOMETRY = "odometry"
    DETECTION = "detection"          # pre-computed 2D/3D boxes (annotated or detector)
    UNKNOWN = "unknown"


class CoordinateFrame(str, Enum):
    """Common reference frames. Explicit transforms are required to change frame."""
    SENSOR = "sensor"
    VEHICLE = "vehicle"
    WORLD = "world"
    ENU = "enu"          # East-North-Up
    NED = "ned"          # North-East-Down
    ECEF = "ecef"
    PIXEL = "pixel"      # image plane – never mix with metric without projection
    UNKNOWN = "unknown"


class TimeUnit(str, Enum):
    SECONDS = "seconds"
    NANOSECONDS = "nanoseconds"
    MILLISECONDS = "milliseconds"


@dataclass
class Timestamp:
    value: float
    unit: TimeUnit = TimeUnit.SECONDS
    source: str = "unknown"

    def to_seconds(self) -> float:
        if self.unit == TimeUnit.SECONDS:
            return float(self.value)
        if self.unit == TimeUnit.NANOSECONDS:
            return float(self.value) * 1e-9
        if self.unit == TimeUnit.MILLISECONDS:
            return float(self.value) * 1e-3
        raise ValueError(f"Unsupported time unit: {self.unit}")


@dataclass
class Pose3D:
    """Rigid transform: position (m) + orientation (quaternion wxyz or rotation matrix)."""
    position: np.ndarray          # shape (3,) meters
    orientation: np.ndarray       # shape (4,) wxyz quaternion or (3,3) rotation matrix
    frame: CoordinateFrame = CoordinateFrame.WORLD
    is_estimate: bool = False     # True if this is filtered / predicted, not direct measurement

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        if self.position.shape != (3,):
            raise ValueError(f"position must be (3,), got {self.position.shape}")
        if self.orientation.shape not in ((4,), (3, 3)):
            raise ValueError(f"orientation must be (4,) or (3,3), got {self.orientation.shape}")


@dataclass
class Observation:
    """Single sensor observation at a given time."""
    modality: SensorModality
    timestamp: Timestamp
    data: Any                                 # np.ndarray, dict, or structured payload
    frame: CoordinateFrame = CoordinateFrame.SENSOR
    sensor_id: str = "default"
    quality: float = 1.0                      # [0, 1]
    is_valid: bool = True
    is_estimate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # numpy arrays are not JSON-serializable by default
        if isinstance(self.data, np.ndarray):
            d["data"] = {
                "shape": list(self.data.shape),
                "dtype": str(self.data.dtype),
                "summary": "ndarray (content omitted in report)",
            }
        return d


@dataclass
class Frame:
    """Synchronized multi-modal snapshot (or loosely associated observations)."""
    frame_id: str
    timestamp: Timestamp
    observations: Dict[SensorModality, Observation] = field(default_factory=dict)
    sequence_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def available_modalities(self) -> List[SensorModality]:
        return [m for m, o in self.observations.items() if o.is_valid]

    def has(self, modality: SensorModality) -> bool:
        return modality in self.observations and self.observations[modality].is_valid


@dataclass
class DataContract:
    """
    Explicit contract that a dataset or stream must satisfy.
    Used by the DataValidator before any downstream processing.
    """
    name: str
    required_modalities: Sequence[SensorModality]
    optional_modalities: Sequence[SensorModality] = field(default_factory=list)
    time_unit: TimeUnit = TimeUnit.SECONDS
    coordinate_frame: CoordinateFrame = CoordinateFrame.WORLD
    min_frames: int = 1
    max_nan_ratio: float = 0.0
    require_finite: bool = True
    require_sorted_timestamps: bool = True
    require_unique_timestamps: bool = True
    max_time_gap_s: Optional[float] = None
    units: Dict[str, str] = field(default_factory=lambda: {
        "position": "m",
        "velocity": "m/s",
        "acceleration": "m/s^2",
        "angular_velocity": "rad/s",
    })
    license: Optional[str] = None
    provenance: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "required_modalities": [m.value for m in self.required_modalities],
            "optional_modalities": [m.value for m in self.optional_modalities],
            "time_unit": self.time_unit.value,
            "coordinate_frame": self.coordinate_frame.value,
            "min_frames": self.min_frames,
            "max_nan_ratio": self.max_nan_ratio,
            "require_finite": self.require_finite,
            "require_sorted_timestamps": self.require_sorted_timestamps,
            "require_unique_timestamps": self.require_unique_timestamps,
            "max_time_gap_s": self.max_time_gap_s,
            "units": self.units,
            "license": self.license,
            "provenance": self.provenance,
            "notes": self.notes,
        }
