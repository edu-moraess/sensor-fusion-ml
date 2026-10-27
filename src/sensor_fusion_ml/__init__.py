"""
sensor-fusion-ml
================

Modular pipeline for multi-sensor fusion, state estimation, multi-object tracking
and trajectory prediction, designed for research and future productization
(ArqTech Labs).

Primary target environment: Google Colab (GPU when available, CPU for tests).

Design principles
-----------------
- Explicit data contracts and validation before any training.
- Clear separation of classical baseline vs. learned models.
- No silent downloads, no automatic heavy training.
- Reproducibility via configs, seeds and versioned checkpoints.
- Missing modalities are first-class citizens.
"""

__version__ = "0.1.0"
__author__ = "ArqTech Labs / Eduardo Moraes"

# Lightweight public API – heavy submodules are imported lazily
from sensor_fusion_ml.data_engine.validator import DataValidator, ValidationReport
from sensor_fusion_ml.data_engine.contracts import DataContract, SensorModality

__all__ = [
    "__version__",
    "DataValidator",
    "ValidationReport",
    "DataContract",
    "SensorModality",
]
