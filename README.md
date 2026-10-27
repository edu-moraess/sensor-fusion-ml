# sensor-fusion-ml

**Modular Sensor Fusion & State Estimation pipeline for robotics and autonomous systems.**

Research-to-product path under ArqTech Labs. Designed to run first on **Google Colab** (GPU when available, CPU for validation).

[![Status](https://img.shields.io/badge/status-Phase%20A%2FC%2FD-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## What is implemented (v0.1.0)

| Component | Status | Notes |
|-----------|--------|-------|
| Data contracts & validator | **PASS** | Explicit modalities, timestamps, finiteness, split leakage |
| Environment audit | **PASS** | No side-effects, JSON reports |
| Constant-velocity Kalman Filter | **PASS** | 2D / 3D, synthetic validation |
| Hungarian / greedy association | **PASS** | |
| MinimalFusionMLP (PyTorch) | **PASS** | Missing-modality embeddings, gradient & checkpoint tests |
| Kinematic trajectory baseline | **PASS** | ADE / FDE ready |
| Unit tests | **PASS** | No network, no GPU, no external data required |
| Real dataset loaders | **NOT RUN** | Intentionally deferred – licensing first |
| Full multi-object tracker | **NOT RUN** | Association only |
| Heavy training on real data | **NOT RUN** | Forbidden until data contracts pass |

---

## Quick start (local or Colab)

```bash
git clone https://github.com/edu-moraess/sensor-fusion-ml.git
cd sensor-fusion-ml

# Recommended: editable install
pip install -e ".[dev]"

# Environment audit (no installs, no downloads)
python scripts/audit_environment.py

# Synthetic data validation
python scripts/validate_data.py --out artifacts/reports/validation.json

# Classical baseline on synthetic trajectory
python scripts/run_baseline.py

# Unit tests
pytest tests/unit -q
```

### Google Colab

Open `notebooks/00_environment_and_audit.ipynb`.  
The notebook:

1. Inspects Python / PyTorch / GPU without installing anything.
2. Explains how to clone and install the package.
3. Runs the audit, synthetic validation and unit tests.
4. Never mounts Drive or downloads datasets automatically.

---

## Repository layout

```
sensor-fusion-ml/
├── configs/                 # YAML configurations
├── docs/                    # Architecture, contracts, licensing, roadmap
├── src/sensor_fusion_ml/    # Installable package
│   ├── data_engine/         # Contracts + Validator + Audit   ← Phase A
│   ├── state_estimation/    # Kalman baseline                ← Phase C
│   ├── tracking/            # Association
│   ├── encoders/            # SmallCNN, SmallPointNet
│   ├── fusion/              # MinimalFusionMLP               ← Phase D
│   ├── trajectory/          # Kinematic baseline             ← Phase E
│   └── evaluation/          # MAE, RMSE, ADE, FDE
├── tests/unit/              # Deterministic, no external data
├── notebooks/               # Colab entry point
├── scripts/                 # CLI entry points
└── artifacts/               # Reports & checkpoints (git-ignored content)
```

---

## Design principles (enforced)

1. **Audit before train** – validation reports with explicit status codes.
2. **No silent downloads** – datasets must be obtained by the user under official licenses.
3. **No invented metrics or results** – un-run experiments are marked `NOT RUN`.
4. **Missing modalities are first-class** – never assume all sensors are present.
5. **Coordinate frames are explicit** – mixing without transforms is a design violation.
6. **Reproducibility** – seed, config, version and data identity travel with every report.

---

## Dataset policy

See [`docs/dataset_licensing.md`](docs/dataset_licensing.md).

- KITTI, nuScenes, Argoverse 2 and Waymo are documented.
- None are downloaded by this repository.
- Synthetic data is used exclusively for software validation.

---

## Next recommended action

1. Obtain a KITTI subset under the official terms.
2. Implement a minimal sequence loader that satisfies the existing `DataContract`.
3. Run the classical baseline on real data and produce a first scientific report (clearly separated from synthetic results).

---

## License

MIT – see [LICENSE](LICENSE).

Copyright (c) 2026 ArqTech Labs / Eduardo Moraes.
