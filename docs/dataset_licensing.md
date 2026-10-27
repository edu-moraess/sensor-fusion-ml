# Dataset Licensing & Access Notes

**Principle:** No dataset is downloaded automatically. Users must obtain data under the official terms and point the configuration to a local path.

## Candidate Datasets

### 1. KITTI Vision Benchmark Suite
- **Official site:** http://www.cvlibs.net/datasets/kitti/
- **License summary:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (non-commercial research). Commercial use requires separate permission.
- **Status in this repo:** Not downloaded. Loader planned after licensing confirmation.

### 2. nuScenes
- **Official site:** https://www.nuscenes.org/
- **License:** Non-commercial research. Commercial use requires separate agreement.
- **Status:** Not downloaded.

### 3. Argoverse 2
- **Official site:** https://www.argoverse.org/av2.html
- **Status:** Not downloaded. Trajectory interfaces and ADE/FDE metrics are ready.

### 4. Waymo Open Dataset
- **Official site:** https://waymo.com/open/
- **Status:** Not downloaded. Not planned for the first experimental phase.

## Synthetic Data

All unit tests and the scripts `validate_data.py` / `run_baseline.py` use only synthetic data generated inside the repository. Synthetic data is for software validation only and must never be presented as scientific evidence of model performance.
