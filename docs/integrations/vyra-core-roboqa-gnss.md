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

The native 2x2 covariance `R` is validated and copied into
`Observation.metadata["measurement_covariance"]` as a JSON-serializable
list. The Observation contract has no first-class covariance field.

## Deliberate limitation

This adapter does not call `ConstantVelocityKF`. That filter currently uses
a configured scalar isotropic measurement-noise matrix and cannot consume an
arbitrary per-observation covariance `R`. A later filter API change must
explicitly accept and validate observation covariance before this data is
used in a Kalman update.

## API

```python
from sensor_fusion_ml.integrations import roboqa_gps_envelope_to_observation

observation = roboqa_gps_envelope_to_observation(vyra_envelope)
```
