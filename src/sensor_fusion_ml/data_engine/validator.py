"""
Data validation engine.

Produces structured ValidationReport objects with explicit status codes.
Never silently drops data; generates evidence for human inspection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .contracts import (
    DataContract,
    Frame,
    Observation,
    SensorModality,
    Timestamp,
    TimeUnit,
)


class ValidationStatus(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


@dataclass
class CheckResult:
    name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Reproducible validation report."""
    contract_name: str
    status: ValidationStatus
    timestamp_utc: str
    checks: List[CheckResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)
        self._recompute_status()

    def _recompute_status(self) -> None:
        statuses = [c.status for c in self.checks]
        if ValidationStatus.BLOCKED in statuses:
            self.status = ValidationStatus.BLOCKED
        elif ValidationStatus.FAIL in statuses:
            self.status = ValidationStatus.FAIL
        elif ValidationStatus.PASS_WITH_WARNINGS in statuses:
            self.status = ValidationStatus.PASS_WITH_WARNINGS
        elif all(s == ValidationStatus.PASS for s in statuses):
            self.status = ValidationStatus.PASS
        else:
            self.status = ValidationStatus.NOT_RUN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "status": self.status.value,
            "timestamp_utc": self.timestamp_utc,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    def __str__(self) -> str:
        lines = [
            f"ValidationReport [{self.status.value}] – {self.contract_name}",
            f"UTC: {self.timestamp_utc}",
            "-" * 60,
        ]
        for c in self.checks:
            lines.append(f"  [{c.status.value:18s}] {c.name}: {c.message}")
        if self.summary:
            lines.append("-" * 60)
            for k, v in self.summary.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class DataValidator:
    """
    Validates frames / sequences against a DataContract.

    Design constraints (from project principles):
    - No automatic data removal.
    - No downloads.
    - Explicit distinction between missing and invalid measurements.
    - Deterministic, reproducible reports.
    """

    def __init__(self, contract: DataContract):
        self.contract = contract

    def validate_frames(
        self,
        frames: Sequence[Frame],
        *,
        sequence_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationReport:
        report = ValidationReport(
            contract_name=self.contract.name,
            status=ValidationStatus.NOT_RUN,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            metadata={
                "sequence_id": sequence_id,
                "n_frames_input": len(frames),
                **(extra_metadata or {}),
            },
        )

        if len(frames) < self.contract.min_frames:
            report.add(CheckResult(
                name="min_frames",
                status=ValidationStatus.FAIL,
                message=f"Only {len(frames)} frames; required >= {self.contract.min_frames}",
                details={"n_frames": len(frames)},
            ))
            return report

        self._check_required_modalities(frames, report)
        self._check_timestamps(frames, report)
        self._check_finite(frames, report)
        self._check_dimensions(frames, report)

        report.summary = {
            "n_frames": len(frames),
            "modalities_seen": sorted({
                m.value for f in frames for m in f.available_modalities()
            }),
            "status": report.status.value,
        }
        return report

    def validate_file_existence(
        self,
        paths: Sequence[Union[str, Path]],
        *,
        must_exist: bool = True,
    ) -> ValidationReport:
        report = ValidationReport(
            contract_name=self.contract.name + "::file_existence",
            status=ValidationStatus.NOT_RUN,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        missing = []
        existing = []
        for p in paths:
            p = Path(p)
            if p.exists() and p.is_file():
                existing.append(str(p))
            else:
                missing.append(str(p))

        if missing and must_exist:
            report.add(CheckResult(
                name="file_existence",
                status=ValidationStatus.FAIL,
                message=f"{len(missing)} required files missing",
                details={"missing": missing[:50], "n_missing": len(missing)},
            ))
        else:
            report.add(CheckResult(
                name="file_existence",
                status=ValidationStatus.PASS,
                message=f"All {len(existing)} files present" if not missing
                        else f"{len(existing)} present, {len(missing)} optional missing",
                details={"n_existing": len(existing), "n_missing": len(missing)},
            ))
        report.summary = {"n_existing": len(existing), "n_missing": len(missing)}
        return report

    def validate_split_integrity(
        self,
        train_ids: Sequence[str],
        val_ids: Sequence[str],
        test_ids: Sequence[str],
    ) -> ValidationReport:
        """Ensure no sequence leakage across splits."""
        report = ValidationReport(
            contract_name=self.contract.name + "::split_integrity",
            status=ValidationStatus.NOT_RUN,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

        leakage = {
            "train_val": sorted(train_set & val_set),
            "train_test": sorted(train_set & test_set),
            "val_test": sorted(val_set & test_set),
        }
        has_leakage = any(leakage.values())

        if has_leakage:
            report.add(CheckResult(
                name="split_leakage",
                status=ValidationStatus.FAIL,
                message="Sequence leakage detected between splits",
                details=leakage,
            ))
        else:
            report.add(CheckResult(
                name="split_leakage",
                status=ValidationStatus.PASS,
                message="No sequence overlap between train/val/test",
                details={
                    "n_train": len(train_set),
                    "n_val": len(val_set),
                    "n_test": len(test_set),
                },
            ))

        for name, s in [("train", train_set), ("val", val_set), ("test", test_set)]:
            if len(s) == 0:
                report.add(CheckResult(
                    name=f"empty_{name}",
                    status=ValidationStatus.PASS_WITH_WARNINGS,
                    message=f"{name} split is empty",
                ))

        report.summary = {
            "n_train": len(train_set),
            "n_val": len(val_set),
            "n_test": len(test_set),
            "has_leakage": has_leakage,
        }
        return report

    def _check_required_modalities(
        self, frames: Sequence[Frame], report: ValidationReport
    ) -> None:
        present = {m for f in frames for m in f.available_modalities()}
        missing = [m for m in self.contract.required_modalities if m not in present]
        if missing:
            report.add(CheckResult(
                name="required_modalities",
                status=ValidationStatus.FAIL,
                message=f"Missing required modalities: {[m.value for m in missing]}",
                details={"missing": [m.value for m in missing],
                         "present": [m.value for m in present]},
            ))
        else:
            report.add(CheckResult(
                name="required_modalities",
                status=ValidationStatus.PASS,
                message="All required modalities observed at least once",
                details={"present": [m.value for m in present]},
            ))

    def _check_timestamps(
        self, frames: Sequence[Frame], report: ValidationReport
    ) -> None:
        ts = [f.timestamp.to_seconds() for f in frames]
        if not ts:
            return

        is_sorted = all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))
        if self.contract.require_sorted_timestamps and not is_sorted:
            report.add(CheckResult(
                name="timestamp_order",
                status=ValidationStatus.FAIL,
                message="Timestamps are not monotonically non-decreasing",
            ))
        else:
            report.add(CheckResult(
                name="timestamp_order",
                status=ValidationStatus.PASS if is_sorted else ValidationStatus.PASS_WITH_WARNINGS,
                message="Timestamps ordered" if is_sorted else "Timestamps not sorted (allowed by contract)",
            ))

        n_unique = len(set(ts))
        if self.contract.require_unique_timestamps and n_unique != len(ts):
            report.add(CheckResult(
                name="timestamp_uniqueness",
                status=ValidationStatus.FAIL,
                message=f"Duplicate timestamps: {len(ts) - n_unique} duplicates",
                details={"n_duplicates": len(ts) - n_unique},
            ))
        else:
            report.add(CheckResult(
                name="timestamp_uniqueness",
                status=ValidationStatus.PASS,
                message="Timestamps unique" if n_unique == len(ts) else "Duplicates allowed by contract",
            ))

        if self.contract.max_time_gap_s is not None and len(ts) > 1:
            gaps = np.diff(ts)
            max_gap = float(np.max(gaps))
            if max_gap > self.contract.max_time_gap_s:
                report.add(CheckResult(
                    name="timestamp_gaps",
                    status=ValidationStatus.PASS_WITH_WARNINGS,
                    message=f"Max gap {max_gap:.4f}s exceeds threshold {self.contract.max_time_gap_s}s",
                    details={"max_gap_s": max_gap, "threshold_s": self.contract.max_time_gap_s},
                ))
            else:
                report.add(CheckResult(
                    name="timestamp_gaps",
                    status=ValidationStatus.PASS,
                    message=f"Max gap {max_gap:.4f}s within limit",
                    details={"max_gap_s": max_gap},
                ))

    def _check_finite(
        self, frames: Sequence[Frame], report: ValidationReport
    ) -> None:
        total_values = 0
        nan_count = 0
        inf_count = 0

        for f in frames:
            for obs in f.observations.values():
                if not obs.is_valid:
                    continue
                arr = self._extract_numeric(obs.data)
                if arr is None:
                    continue
                total_values += arr.size
                nan_count += int(np.isnan(arr).sum())
                inf_count += int(np.isinf(arr).sum())

        if total_values == 0:
            report.add(CheckResult(
                name="finite_values",
                status=ValidationStatus.PASS_WITH_WARNINGS,
                message="No numeric arrays found to check finiteness",
            ))
            return

        nan_ratio = nan_count / total_values
        if self.contract.require_finite and (nan_count > 0 or inf_count > 0):
            # Infinity is never covered by the NaN tolerance; it is always invalid when finiteness is required.
            status = (
                ValidationStatus.FAIL
                if inf_count > 0 or nan_ratio > self.contract.max_nan_ratio
                else ValidationStatus.PASS_WITH_WARNINGS
            )
            report.add(CheckResult(
                name="finite_values",
                status=status,
                message=f"NaN={nan_count}, Inf={inf_count} (ratio={nan_ratio:.6f})",
                details={"nan_count": nan_count, "inf_count": inf_count, "total": total_values},
            ))
        else:
            report.add(CheckResult(
                name="finite_values",
                status=ValidationStatus.PASS,
                message="All checked values finite" if nan_count == 0 and inf_count == 0
                        else f"NaN ratio {nan_ratio:.6f} within tolerance",
                details={"nan_count": nan_count, "inf_count": inf_count, "total": total_values},
            ))

    def _check_dimensions(
        self, frames: Sequence[Frame], report: ValidationReport
    ) -> None:
        """Lightweight consistency check: same modality should have compatible shapes."""
        shapes: Dict[SensorModality, List[Tuple]] = {}
        for f in frames:
            for m, obs in f.observations.items():
                if not obs.is_valid:
                    continue
                arr = self._extract_numeric(obs.data)
                if arr is not None:
                    shapes.setdefault(m, []).append(arr.shape)

        inconsistent = []
        for m, sh_list in shapes.items():
            unique = set(sh_list)
            if len(unique) > 1:
                if m == SensorModality.RGB:
                    inconsistent.append((m.value, list(unique)[:5]))

        if inconsistent:
            report.add(CheckResult(
                name="dimension_consistency",
                status=ValidationStatus.PASS_WITH_WARNINGS,
                message="Some modalities have varying shapes",
                details={"inconsistent": inconsistent},
            ))
        else:
            report.add(CheckResult(
                name="dimension_consistency",
                status=ValidationStatus.PASS,
                message="No critical dimension inconsistencies detected",
            ))

    @staticmethod
    def _extract_numeric(data: Any) -> Optional[np.ndarray]:
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    return v
        return None
