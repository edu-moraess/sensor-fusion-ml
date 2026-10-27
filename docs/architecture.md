# Architecture – sensor-fusion-ml

**Status:** Phase A (Data Engine) + Phase C (classical baseline) + Phase D (minimal trainable model) implemented.  
Higher phases are planned and interfaces are reserved.

## Design Goals

1. Explicit data contracts before any learning.
2. Classical baseline that is independently evaluable.
3. Trainable models that never silently mix coordinate frames.
4. First-class support for missing modalities.
5. Colab-first, CPU-testable, GPU-acceleratable.
6. No automatic downloads or heavy training on import.

## Module Map

| Package | Responsibility | Status |
|---------|----------------|--------|
| `data_engine` | Contracts, validation, audit, reports | **Implemented** |
| `sensor_io` | Loaders for real datasets | Planned (interfaces only) |
| `time_sync` | Multi-rate temporal alignment | Planned |
| `calibration` | Intrinsics / extrinsics | Planned |
| `encoders` | Visual & point-cloud feature extractors | Minimal CNN + PointNet implemented |
| `fusion` | Multi-modal fusion networks | `MinimalFusionMLP` implemented |
| `state_estimation` | Kalman & classical filters | `ConstantVelocityKF` implemented |
| `tracking` | Data association | Hungarian + greedy implemented |
| `trajectory` | Prediction interfaces & kinematic baseline | Implemented |
| `evaluation` | Metrics with GT requirements | MAE/RMSE/ADE/FDE implemented |
| `inference` | Serving & latency | Planned |

## Known Limitations (current delivery)

- No real dataset loaders yet (by design – licensing must be reviewed first).
- No multi-object tracker full lifecycle (birth/death beyond association).
- No uncertainty calibration or probabilistic heads yet.
- Trajectory horizon is fixed at construction time.
- Visual encoder is intentionally tiny (software validation only).
