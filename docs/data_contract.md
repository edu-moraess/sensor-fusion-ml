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
