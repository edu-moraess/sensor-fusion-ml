# Evaluation Protocol

## Rules

1. Metrics are computed **only** when valid ground-truth references exist for the evaluated quantity.
2. Training loss is never reported as a substitute for test-set performance.
3. Hyper-parameters are tuned exclusively on the validation split.
4. Test sequences are never used for model selection.
5. Splits are formed by **sequence / scene**, never by random frame shuffling.
6. Every report records: dataset (or “synthetic”), split, config hash, checkpoint id, units, number of samples.
7. Unit-test pass/fail results are **not** performance metrics and must never be presented as such.

## A. Implemented Metrics

The following metrics are present in `src/sensor_fusion_ml/evaluation/metrics.py` and covered by unit tests. They measure localisation or trajectory error; they do **not** evaluate identity consistency or multi-object association quality.

| Metric | Quantity | Expected input shape | Unit (conventional) |
|--------|----------|----------------------|---------------------|
| MAE | Mean absolute error (element-wise) | matching shapes | same as input |
| RMSE | Root mean square error | matching shapes | same as input |
| ADE | Average Displacement Error – mean L2 over all future timesteps | `(..., T, D)` | same as position |
| FDE | Final Displacement Error – L2 at the last predicted step | `(..., T, D)` | same as position |

Notes:

- The functions accept NumPy arrays or torch tensors (converted via `.detach().cpu().numpy()`).
- Shape mismatch raises `ValueError`.
- Conventional units shown in older documentation (metres, m/s) are **not enforced** by the metric functions; the caller is responsible for supplying quantities in a consistent unit system.
- Velocity MAE/RMSE can be obtained by calling the same functions on velocity arrays; there is no dedicated velocity metric implementation.

## B. Planned MOT Metrics (not implemented)

The metrics below are required for a proper multi-object tracking evaluation. They are **not present** in the codebase and must not be reported as available.

| Metric | Purpose |
|--------|---------|
| MOTA | Multi-Object Tracking Accuracy (accounts for FP, FN, ID switches) |
| IDF1 | Identification F1 — measures identity consistency via bipartite matching between predicted and ground-truth trajectories over a sequence |
| HOTA | Higher-Order Tracking Accuracy — combines detection and association quality; requires a documented matching protocol (distance/IoU threshold) |
| ID Switches | Number of identity switches |
| Fragmentation | Number of track fragmentations |
| False Positives | Detections without corresponding ground-truth |
| False Negatives | Ground-truth objects without a corresponding track |

None of the metrics in this table are implemented in the current codebase.

Any future implementation must be accompanied by:

- Explicit ground-truth format with persistent object identities.
- A documented matching procedure (distance or IoU threshold).
- Clear treatment of frames that contain no detections or no ground-truth objects.
- Separation between localisation error and identity error.

## C. Requirements for Future MOT Evaluation

A reproducible MOT evaluation protocol for this repository shall include at minimum:

1. **Ground truth**
   - Sequence of frames with object positions and persistent integer identities.
   - Explicit handling of objects that appear, disappear or are occluded.

2. **Temporal alignment**
   - Predictions and ground truth must share a common, ordered timeline.

3. **Matching definition**
   - Distance (or IoU) threshold used to decide whether a predicted track matches a ground-truth object.
   - The threshold must be documented and fixed for a given experiment.

4. **Identity metrics**
   - Computation of ID switches, fragmentation, IDF1 / HOTA as appropriate.

5. **Localisation metrics**
   - Position or bounding-box error computed only on matched pairs (distinct from identity metrics).

6. **Edge cases**
   - Frames with zero detections.
   - Frames with zero ground-truth objects.
   - Tracks that are initialized but never confirmed (if semantic states are later introduced).

7. **Reporting**
   - Dataset (or “synthetic”), split, configuration hash, matching threshold, number of sequences and frames.
   - Clear labelling of results obtained on synthetic data as “software validation only”.

Until the infrastructure above exists, only the localisation metrics listed in section A may be reported, and only when genuine ground-truth trajectories are supplied.

## Synthetic vs Real

- Results obtained on synthetic data are labelled **“software validation only”**.
- They must not be cited as evidence of real-world accuracy.
