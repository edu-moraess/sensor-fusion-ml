# Data Contract Specification

## Purpose

A `DataContract` is a machine-readable declaration of what a dataset or stream must satisfy before any training or evaluation proceeds.

## Status Codes

| Status | Meaning |
|--------|---------|
| `PASS` | All checks succeeded |
| `PASS_WITH_WARNINGS` | Usable but non-critical issues found |
| `FAIL` | Contract violation – do not train/evaluate |
| `BLOCKED` | Prerequisite missing (e.g. CUDA required but absent) |
| `NOT_RUN` | Check was not executed |

## Validation Guarantees

- No data is deleted or silently repaired.
- Duplicate or out-of-order timestamps produce evidence, not automatic drops.
- Split leakage is reported as `FAIL`.
- Reports are JSON-serializable and timestamped (UTC).


## Observation quality semantics

`Observation.quality` is an optional numeric score in the closed interval
`[0, 1]`. A value of `None` means that the source did not provide a
comparable numeric quality score. It must not be replaced with `1.0` merely
because the source artifact is valid or because the adapter successfully
parsed it.

When present, quality must be finite and numeric. Booleans, strings, NaN,
infinity, and values outside `[0, 1]` are rejected. Detector confidence,
categorical source-quality labels, and measurement covariance are distinct
quantities and must remain in their own fields or metadata.
