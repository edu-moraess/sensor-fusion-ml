"""Local manifest-based sequence loader.

This module performs no network access, downloads, installation, or training.
Every payload path must resolve inside the explicitly supplied dataset root.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    CoordinateFrame,
    DataContract,
    Frame,
    Observation,
    SensorModality,
    TimeUnit,
    Timestamp,
)
from .validator import DataValidator, ValidationReport, ValidationStatus


_SUPPORTED_FORMATS = {"npy", "json", "csv", "path"}


@dataclass
class LoadedDataset:
    """Loaded sequences plus source provenance and declared split membership."""

    name: str
    license: str
    provenance: str
    root: Path
    sequences: dict[str, list[Frame]] = field(default_factory=dict)
    splits: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _enum_value(enum_type, value: str, field_name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = [item.value for item in enum_type]
        raise ValueError(f"Invalid {field_name}={value!r}; expected one of {allowed}") from exc


def _safe_payload_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Payload path escapes dataset root: {relative_path!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Payload file does not exist: {relative_path!r}")
    return candidate


def _load_payload(path: Path, payload_format: str) -> Any:
    if payload_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported payload format {payload_format!r}; "
            f"supported formats: {sorted(_SUPPORTED_FORMATS)}"
        )
    if payload_format == "npy":
        return np.load(path, allow_pickle=False)
    if payload_format == "json":
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    if payload_format == "csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        try:
            return np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"CSV payload must contain numeric values only: {path.name}") from exc
    # Path mode intentionally returns a reference; it does not decode image/video bytes.
    return str(path)


def load_manifest(manifest_path: str | Path, *, data_root: str | Path | None = None) -> LoadedDataset:
    """Load a local JSON manifest into Frames, without downloading or dropping records.

    Manifest paths are relative to data_root (or the manifest's parent when omitted).
    Supported payload formats: npy, json, csv, path. The path format returns a local
    file reference, useful for image/video assets that a modality-specific decoder may
    consume later.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    root = Path(data_root).expanduser().resolve() if data_root is not None else manifest_path.parent
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object")

    required_top = ("name", "license", "provenance", "time_unit", "sequences")
    missing = [key for key in required_top if key not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")
    if not str(manifest["license"]).strip() or not str(manifest["provenance"]).strip():
        raise ValueError("Manifest license and provenance must be non-empty")

    time_unit = _enum_value(TimeUnit, manifest["time_unit"], "time_unit")
    dataset = LoadedDataset(
        name=str(manifest["name"]),
        license=str(manifest["license"]),
        provenance=str(manifest["provenance"]),
        root=root,
        metadata=manifest.get("metadata", {}),
    )

    seen_sequence_ids: set[str] = set()
    seen_split_ids: dict[str, set[str]] = {}
    sequence_entries = manifest["sequences"]
    if not isinstance(sequence_entries, list) or not sequence_entries:
        raise ValueError("Manifest 'sequences' must be a non-empty list")

    for seq_entry in sequence_entries:
        if not isinstance(seq_entry, dict):
            raise ValueError("Each sequence entry must be an object")
        sequence_id = str(seq_entry.get("sequence_id", "")).strip()
        split = str(seq_entry.get("split", "")).strip().lower()
        frames_entry = seq_entry.get("frames")
        if not sequence_id:
            raise ValueError("Every sequence requires a non-empty sequence_id")
        if sequence_id in seen_sequence_ids:
            raise ValueError(f"Duplicate sequence_id: {sequence_id}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Sequence {sequence_id}: split must be train, val, or test")
        if not isinstance(frames_entry, list) or not frames_entry:
            raise ValueError(f"Sequence {sequence_id}: frames must be a non-empty list")

        frames: list[Frame] = []
        for frame_entry in frames_entry:
            if not isinstance(frame_entry, dict):
                raise ValueError(f"Sequence {sequence_id}: each frame must be an object")
            frame_id = str(frame_entry.get("frame_id", "")).strip()
            if not frame_id or "timestamp" not in frame_entry:
                raise ValueError(f"Sequence {sequence_id}: frame requires frame_id and timestamp")
            frame_timestamp = Timestamp(
                value=float(frame_entry["timestamp"]),
                unit=time_unit,
                source=str(frame_entry.get("timestamp_source", "manifest")),
            )
            observations: dict[SensorModality, Observation] = {}
            obs_entries = frame_entry.get("observations", [])
            if not isinstance(obs_entries, list):
                raise ValueError(f"Frame {frame_id}: observations must be a list")
            for obs_entry in obs_entries:
                modality = _enum_value(SensorModality, obs_entry.get("modality"), "modality")
                if modality in observations:
                    raise ValueError(
                        f"Frame {frame_id}: duplicate modality {modality.value}; "
                        "Frame currently supports one observation per modality"
                    )
                if "path" not in obs_entry or "format" not in obs_entry:
                    raise ValueError(f"Frame {frame_id}, {modality.value}: path and format required")
                payload_path = _safe_payload_path(root, str(obs_entry["path"]))
                payload = _load_payload(payload_path, str(obs_entry["format"]).lower())
                obs_time = Timestamp(
                    value=float(obs_entry.get("timestamp", frame_entry["timestamp"])),
                    unit=_enum_value(
                        TimeUnit,
                        obs_entry.get("time_unit", time_unit.value),
                        "observation time_unit",
                    ),
                    source=str(obs_entry.get("timestamp_source", "manifest")),
                )
                observations[modality] = Observation(
                    modality=modality,
                    timestamp=obs_time,
                    data=payload,
                    frame=_enum_value(
                        CoordinateFrame, obs_entry.get("frame", "unknown"), "coordinate frame"
                    ),
                    sensor_id=str(obs_entry.get("sensor_id", "default")),
                    quality=float(obs_entry.get("quality", 1.0)),
                    is_valid=bool(obs_entry.get("is_valid", True)),
                    is_estimate=bool(obs_entry.get("is_estimate", False)),
                    metadata=obs_entry.get("metadata", {}),
                )
            frames.append(Frame(
                frame_id=frame_id,
                timestamp=frame_timestamp,
                observations=observations,
                sequence_id=sequence_id,
                metadata=frame_entry.get("metadata", {}),
            ))

        seen_sequence_ids.add(sequence_id)
        seen_split_ids.setdefault(split, set()).add(sequence_id)
        dataset.sequences[sequence_id] = frames

    # Sequence IDs are unique by construction, and each sequence has exactly one split.
    dataset.splits = {key: sorted(value) for key, value in seen_split_ids.items()}
    return dataset


def validate_dataset(
    dataset: LoadedDataset, contract: DataContract
) -> dict[str, ValidationReport]:
    """Validate each loaded sequence and enforce sequence-level split integrity."""
    validator = DataValidator(contract)
    reports = {
        sequence_id: validator.validate_frames(frames, sequence_id=sequence_id, extra_metadata={
            "dataset_name": dataset.name,
            "license": dataset.license,
            "provenance": dataset.provenance,
            "split": next(split for split, ids in dataset.splits.items() if sequence_id in ids),
        })
        for sequence_id, frames in dataset.sequences.items()
    }
    train_ids = dataset.splits.get("train", [])
    val_ids = dataset.splits.get("val", [])
    test_ids = dataset.splits.get("test", [])
    split_report = validator.validate_split_integrity(train_ids, val_ids, test_ids)
    reports["__split_integrity__"] = split_report
    return reports


def validation_passed(reports: dict[str, ValidationReport]) -> bool:
    """True only when every report is PASS or PASS_WITH_WARNINGS."""
    return bool(reports) and all(
        report.status in (ValidationStatus.PASS, ValidationStatus.PASS_WITH_WARNINGS)
        for report in reports.values()
    )
