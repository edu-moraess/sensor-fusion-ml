# Development Roadmap

## Phase A – Data Engine (current)
- [x] Contracts (`SensorModality`, `Observation`, `Frame`, `DataContract`)
- [x] Validator with status codes and JSON reports
- [x] Environment audit
- [x] Split leakage detection
- [x] Unit tests

## Phase B – Sensor contracts & geometry
- [ ] Concrete loaders for KITTI (after licensing confirmation)
- [ ] Timestamp multi-rate synchronization utilities
- [ ] Extrinsic / intrinsic calibration containers

## Phase C – Classical baseline
- [x] Constant-velocity Kalman Filter (2D / 3D)
- [x] Hungarian + greedy association
- [x] Synthetic end-to-end script
- [ ] Full track management (birth / death / confirmation)

## Phase D – Minimal trainable model
- [x] SmallCNN + SmallPointNet
- [x] MinimalFusionMLP with missing-modality embeddings
- [x] Gradient and checkpoint unit tests

## Phase E – Trajectory prediction
- [x] Kinematic constant-velocity baseline
- [x] ADE / FDE metrics

## Immediate next recommended action
1. Obtain KITTI odometry or tracking subset under the official non-commercial terms.
2. Implement a minimal KITTI sequence loader that respects the DataContract.
3. Run the classical baseline on a real sequence and produce a first scientific report (still keeping synthetic results clearly separated).
