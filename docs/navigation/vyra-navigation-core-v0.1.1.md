# Vyra Navigation Core v0.1.1

Initial mathematical foundation for local ENU inertial navigation and a
15-dimensional error-state Kalman filter (ESKF).

## Included

- Hamilton scalar-first quaternions and body-to-ENU rotation.
- Nominal strapdown propagation with midpoint attitude for specific force.
- 15D error-state continuous matrices using a right-multiplicative,
  body-frame attitude error.
- First-order covariance discretization.
- Joseph-form GNSS position and ZUPT velocity updates.
- Deterministic unit tests and the Colab validation report.

## Validation evidence

The Colab execution supplied for this baseline reported **10 tests passed,
0 failures, 0 errors** (0.379 s). The report is preserved at
`reports/vyra_navigation_core_v0_1_1_report.json`.

## Important limitations

This is a mathematical/software baseline, not a validated navigation solution.
Noise defaults are placeholders and are not PPC sensor calibration values.
No real dataset was processed by this package. No RINEX positioning was run.
The GNSS update assumes the observation is already in the same ENU frame and
origin as the state. No antenna lever-arm model is included. ZUPT must only
be invoked after an independent stationary detector. Covariance propagation
uses first-order discretization.

## Run tests

From the repository root with the project test dependencies installed:

```bash
pytest -q tests/unit/test_vyra_navigation_core.py
```
