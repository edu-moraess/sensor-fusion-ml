# Vyra Core — RoboQA GNSS consumer

## Scope

This optional integration consumes a Vyra Core envelope whose payload is a
native RoboQA GPS Measurement. It creates a Sensor Fusion ML
`Observation(modality=GNSS)`. The Sensor Fusion ML repository remains
independently usable; neither RoboQA nor Vyra Core is a runtime dependency.

## Accepted spatial semantics

The consumer accepts only:

- `coordinate_frame="LOCAL_CARTESIAN"`
- `axis_order=("x", "y")`
- `units=("m", "m")`
- explicit `coordinate_system`, `crs`, and `origin`

This matches the RoboQA KITTI path: latitude/longitude is converted to local
Cartesian coordinates using the documented equirectangular approximation,
with the first OXTS sample as origin. Synthetic RoboQA trajectories are also
local Cartesian metres, but have a synthetic origin. The caller must identify
which coordinate system and origin apply in the envelope.

Geographic latitude/longitude, UTM assumptions, and implicit transforms are
rejected. The adapter does not transform coordinates.

## Time and covariance

The envelope must declare numeric event time in seconds, an explicit
`time_reference`, and `clock_domain`. The event time must equal the native
Measurement timestamp. KITTI OXTS timestamps in the current RoboQA loader are
approximated at 10 Hz; callers must not label these as UTC or synchronized
wall-clock time.

The native 2x2 covariance `R` is validated and represented as the
first-class `Observation.uncertainty` contract:

- status: `provided`
- covariance: native 2x2 matrix
- variable order: `("x", "y")`
- units: `("m", "m")`
- frame: `LOCAL_CARTESIAN`
- provenance: source Vyra artifact ID

For transition compatibility, the adapter also retains the legacy
`metadata["measurement_covariance"]` JSON list. The typed uncertainty field
is authoritative; consumers must not infer covariance semantics from arbitrary
metadata.

## Deliberate limitation

This adapter does not call `ConstantVelocityKF`. That filter currently uses
a configured scalar isotropic measurement-noise matrix and cannot consume an
arbitrary per-observation covariance `R`. A later filter API change must explicitly accept and validate the typed
observation covariance before this data is used in a Kalman update. This
F-01 contract change does not alter either estimator's update equations.

## API

```python
from sensor_fusion_ml.integrations import roboqa_gps_envelope_to_observation

observation = roboqa_gps_envelope_to_observation(vyra_envelope)
```
