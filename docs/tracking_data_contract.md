# Tracking Data Contract

**Scope:** Formal contracts derived exclusively from the current implementation of
`TrackManager` (`src/sensor_fusion_ml/tracking/manager.py`),
`ConstantVelocityKF` / `KalmanState` (`src/sensor_fusion_ml/state_estimation/kalman.py`) and
`hungarian_association` (`src/sensor_fusion_ml/tracking/associator.py`).

This document describes behaviour that is verified by code and unit tests.
Items that are not defined in the code are explicitly marked as **not formalized**.

---

## 1. Detection Contract

### Current API (TrackManager.update)

| Aspect | Contract |
|--------|----------|
| Type | Numeric array convertible by `np.asarray` |
| Shape | `(N, dim)` where `dim` is the value configured at `TrackManager(dim=...)` construction (must be 2 or 3) |
| Dtype | Converted internally to `float64`. Boolean, complex, object, string and non-real kinds are rejected |
| Values | Every element must be finite (`np.isfinite`). NaN and ±Inf raise `ValueError` |
| Meaning | Each row is treated as a position measurement in the same Cartesian space used by the Kalman filter |
| Empty input | `N = 0` (empty array of correct shape) is valid and produces no new tracks |

### Pre-conditions for association

- Detections and predicted track positions must share the same dimensionality.
- Association cost is Euclidean distance in that space.
- A hard gate (`max_cost`) discards pairs whose distance exceeds the threshold before assignment.

### What is not part of the current detection contract

- Bounding-box representation, confidence scores, class labels, sensor IDs or covariance of the measurement.
- Image-plane coordinates, pixel units or any projective quantity.
- Geographic or geodetic coordinates.

Any richer detection schema must be reduced to a plain `(N, dim)` position array before calling `TrackManager.update`.

---

## 2. Coordinate Contract

### Verified behaviour

- The TrackManager and ConstantVelocityKF operate on an abstract Cartesian vector space of dimension 2 or 3.
- Position and velocity components occupy contiguous blocks of the state vector:
  - 2D: `[x, y, vx, vy]`
  - 3D: `[x, y, z, vx, vy, vz]`
- Observation matrix `H` extracts only the position block.
- Association cost is the Euclidean norm of the difference between predicted position and detection.

### Explicit statements

> The TrackManager expects that detections and filter states are represented in the **same Cartesian reference frame** and are dimensionally compatible.

> Transformations between image coordinates, LiDAR coordinates, world coordinates, body coordinates or geographic coordinates **must occur in an external layer** before association and filtering.

### Not formalized in code

| Item | Status |
|------|--------|
| Origin of the Cartesian frame | Not defined |
| Axis orientation / handedness | Not defined |
| Physical unit of length | Not defined (no conversion factors exist) |
| Physical unit of velocity | Derived from length unit and timestamp unit; not independently declared |
| Relationship to any particular sensor frame | None |

Until these quantities are formalized by a higher-level contract or configuration, callers are responsible for supplying mutually consistent numerical values.

The geometry module (`geometry/projection.py`) provides validated LiDAR-to-camera projection but is **not** consumed by TrackManager.

---

## 3. Temporal Contract

### Accepted timestamp values

- Must be convertible to a finite Python `float`.
- Non-finite values (`NaN`, `+Inf`, `-Inf`) raise `ValueError`.

### Ordering rules

| Situation | Behaviour |
|-----------|-----------|
| First call (`_last_timestamp is None`) | Accepted; becomes the reference |
| `timestamp == _last_timestamp` | Accepted (zero or near-zero `dt`) |
| `timestamp < _last_timestamp` | Rejected with `ValueError("timestamp cannot regress")` |
| `timestamp > _last_timestamp` | Accepted; `dt` is used for prediction |

### Delta-time and prediction

- `dt = current_timestamp - state.timestamp`
- The transition matrix `F` and process-noise matrix `Q` are functions of the real `dt`.
- Negative `dt` inside the Kalman filter itself also raises `ValueError("Cannot predict backwards in time")`.

### First frame

- No prior tracks exist.
- Every valid detection produces a new track via `initiate`.
- `time_since_update` starts at 0, `hits` starts at 1, `age` starts at 1.

### Not formalized

- Absolute time base (Unix epoch, monotonic clock, sensor time, etc.).
- Required units of the timestamp (seconds, milliseconds, \ldots). Callers must keep a consistent unit across a sequence.
- Behaviour under extremely large `dt` (numerical stability is protected by finite-value checks, but no domain-specific limit is imposed).

---

## 4. Track Lifecycle Contract

### Identity

- Track IDs are non-negative integers assigned monotonically by an internal counter (`_next_track_id`).
- IDs are never reused after expiration within a single manager lifetime.
- `reset()` clears all tracks and restarts the counter at 0.

### Counters maintained on every KalmanState

| Field | Meaning | Updated when |
|-------|---------|--------------|
| `hits` | Number of successful measurement updates | Starts at 1 on `initiate`; incremented on every `update` |
| `age` | Number of prediction steps the track has lived (including initiation) | Starts at 1 on `initiate`; incremented on every `predict`; **unchanged** on `update` |
| `time_since_update` | Number of consecutive predictions without a measurement | Starts at 0 on `initiate`; set to 0 on `update`; incremented on every `predict` |

### Lifecycle events (verified)

1. **Creation** – unmatched detection → `kf.initiate(..., track_id=next_id)`.
2. **Prediction** – every active track is predicted to the current timestamp before association.
3. **Association** – gated Hungarian (or fallback) produces matches, unmatched tracks and unmatched detections.
4. **Update** – matched tracks receive a measurement update; `hits += 1`, `time_since_update = 0`.
5. **Aging / Expiration** – an unmatched track is retained while `time_since_update <= max_age`; otherwise it is discarded.
6. **Ordering** – returned list is sorted by ascending `track_id`.
7. **Snapshot isolation** – `tracks` property and the return value of `update` are deep copies; mutation of the returned objects does not affect internal state.
8. **Atomicity** – if any validation, prediction, association validation or filter step raises, the manager’s internal state remains unchanged.

### Semantic states not present

The implementation does **not** maintain explicit flags or enums for:

- `tentative`
- `confirmed`
- `deleted`

Any policy that requires those notions must be built on top of the existing counters (`hits`, `age`, `time_since_update`) by the caller.

### Reset

`TrackManager.reset()` removes all tracks, sets `_next_track_id = 0` and clears `_last_timestamp`.

---

## 5. Association Contract (summary)

- Input: two arrays of shape `(N_tracks, D)` and `(N_detections, D)`.
- Cost matrix: Euclidean distances.
- Gate: pairs with distance > `max_cost` are forbidden.
- Output: list of `(track_idx, detection_idx)` pairs (sorted), plus lists of unmatched indices.
- One-to-one matching is enforced both by the solver and by a post-validation step inside TrackManager.
- Empty inputs are handled deterministically.

---

## 6. Future extensions (out of scope of this contract)

The following are intentionally left for later phases and must not be assumed to exist:

- Rich detection objects (bbox, score, class, sensor metadata).
- Explicit track state machine (`tentative` → `confirmed` → `deleted`).
- Built-in coordinate-frame conversions.
- MOT evaluation metrics (see `docs/evaluation_protocol.md`).
- A higher-level TrackingPipeline that wires detectors to TrackManager.
