# GNSS 2D horizontal update — Stage 4.4 / F-04

## Contract

The mapping identifier `gnss_2d_horizontal_v1` defines a two-dimensional
measurement vector in metres:

`z = [East, North]`

The ESKF error state is
`[dp, dv, dtheta_body, dba, dbg]` with 15 components. The measurement
Jacobian is 2x15 and selects only the East and North position-error components:

- `H[0, 0] = 1` (East)
- `H[1, 1] = 1`
- all other entries are zero.

The residual is `z - state.p[0:2]`; the supplied covariance must be a finite,
symmetric, positive-definite 2x2 matrix. The existing Joseph-form measurement
update and right-multiplicative error injection are reused.

## Reference safety

The caller must provide both `measurement_reference_id` and
`state_reference_id`. They must be non-empty and exactly equal. This check
does not transform coordinates or prove the truth of a caller's declaration;
it prevents accidental use when the two declared local origins differ.

The input must already be East/North in the ESKF's local ENU horizontal
reference. The existing RoboQA `LOCAL_CARTESIAN` observation is not silently
treated as ENU. A future adapter must establish axis semantics, transform
provenance, and origin equivalence explicitly before calling this function.

## Vertical observability

The update directly observes East and North only. The report records
`vertical_measurement_status = NOT_OBSERVED`.

The vertical state and its variance remain unchanged when the prior covariance
has no relevant cross-correlation. If prior cross-covariances couple vertical
position to observed states, the Kalman update may change the vertical estimate
and variance indirectly. Such an indirect correction is not an altitude
measurement; the report marks it as `cross_covariance_only`.

## Non-claims

- No altitude is inferred from the 2D measurement.
- No coordinate transform is performed.
- No local-origin equivalence is established by this function; matching IDs
  are caller-supplied evidence, not a geodetic validation.
- No real GNSS/IMU experiment or scientific navigation validation is implied.
- Existing 3D GNSS update behavior is unchanged.
