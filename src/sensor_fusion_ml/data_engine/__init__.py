"""
Data Engine – Phase A
=====================

Responsible for:
- File existence and integrity checks
- Metadata, timestamp, dimension and finiteness validation
- Provenance and licensing registration
- Label and unit validation
- Duplicate and temporal-order detection
- Split integrity verification
- Reproducible JSON reports with status: PASS | PASS_WITH_WARNINGS | FAIL | BLOCKED | NOT_RUN
"""

from .contracts import DataContract, SensorModality, Observation, Frame
from .validator import DataValidator, ValidationReport, ValidationStatus
from .audit import audit_environment, EnvironmentReport

__all__ = [
    "DataContract",
    "SensorModality",
    "Observation",
    "Frame",
    "DataValidator",
    "ValidationReport",
    "ValidationStatus",
    "audit_environment",
    "EnvironmentReport",
]
