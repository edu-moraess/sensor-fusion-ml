# PPC Frame Audit v1.2 — Engineering Verification

**Project:** ARQTECH LABS / VYRA  
**Repository:** `edu-moraess/sensor-fusion-ml`  
**Execution date:** 2026-11-10  
**Dataset:** PPC Dataset — Nagoya / run1  
**Status:** `FRAME_AUDIT_REQUIRES_REVIEW`

## 1. Purpose

Preserve the results of the PPC Frame Audit v1.2 executed in Google Colab. The audit checks dataset schema, temporal alignment, numerical validity of the candidate rotation matrices, and agreement between measured IMU channels and quantities derived from the reference trajectory.

This is a diagnostic audit, not a physical navigation validation.

## 2. Dataset and temporal integrity

| Check | Result |
|---|---:|
| IMU shape | 153,001 × 8 |
| Reference shape | 7,651 × 14 |
| IMU frequency | 100.0001 Hz |
| Reference frequency | 5.0000 Hz |
| Common interval | 1,530.00 s |
| Column mapping | PASS |

## 3. Rotation-matrix numerical checks

| Check | Result |
|---|---:|
| Maximum orthogonality error | 5.515410169596886e-16 |
| Maximum determinant error | 6.661338147750939e-16 |
| Numerical SO(3) check | PASS |

These results establish that the constructed matrices are numerically valid rotation matrices. They do **not** establish that the selected Euler convention, heading sign, or physical frame mapping matches the PPC reference system.

## 4. Raw IMU statistics

| Statistic | X | Y | Z |
|---|---:|---:|---:|
| Acceleration mean (m/s²) | 0.290201 | -0.243898 | 9.797604 |
| Acceleration std (m/s²) | 0.493960 | 0.325441 | 0.403032 |
| Gyroscope mean (rad/s) | 0.000614 | 0.006652 | 0.000217 |
| Gyroscope std (rad/s) | 0.012372 | 0.014142 | 0.058547 |

Additional statistics:
- Median acceleration norm: 9.8100317 m/s²
- P95 acceleration norm: 10.3714854 m/s²
- Median gyroscope norm: 0.00859348 rad/s

## 5. Accelerometer comparison

The expected body-frame specific force was derived from the 5 Hz reference velocity differentiated over time, with gravity applied in ENU and the candidate body-to-ENU rotation used to transform the result into the body frame.

| Axis | Correlation | Measured-vs-expected slope | Bias (m/s²) | RMSE (m/s²) | Residual std (m/s²) |
|---|---:|---:|---:|---:|---:|
| X | 0.961309 | 1.004099 | -0.094013 | 0.165336 | 0.136005 |
| Y | -0.838320 | -1.007733 | -0.396648 | 0.695704 | 0.571554 |
| Z | -0.444321 | -1.610251 | 19.591515 | 19.596444 | 0.439502 |

Interpretation:
- X shows strong agreement under the tested convention.
- Y shows strong inverse association, but this alone does not prove a physical axis inversion.
- Z has a large systematic discrepancy, with a bias near +19.59 m/s². This is consistent with a major gravity/sign/frame-convention mismatch, but does not uniquely identify its cause.

## 6. Gyroscope comparison

Reference angular rates were derived from consecutive reference attitude rotations at 5 Hz. They are therefore bandwidth-limited and sensitive to the reference Euler convention, attitude noise, and discrete differentiation.

| Axis | Correlation | Measured-vs-expected slope | Bias (rad/s) | RMSE (rad/s) | Residual std (rad/s) |
|---|---:|---:|---:|---:|---:|
| X | 0.848169 | 0.796901 | 0.000354 | 0.005553 | 0.005542 |
| Y | -0.888526 | -0.743410 | 0.006669 | 0.021737 | 0.020688 |
| Z | -0.999539 | -0.999422 | 0.003268 | 0.116843 | 0.116798 |

The inverse Z association is strong evidence of a convention disagreement in the tested comparison, but it is not sufficient by itself to authorize a sign flip.

## 7. Lever arm

The Nagoya lever-arm value recorded for follow-up is:

`[0.593, -0.670, -1.216] m`

It was **not applied** in v1.2. The physical origin represented by `reference.csv` must first be confirmed (IMU origin, GNSS antenna, or another reference point).

## 8. Limitations and interpretation

1. The Euler-angle convention and heading definition used by the audit remain hypotheses pending independent confirmation from the PPC/Applanix documentation.
2. Numerical SO(3) validity does not establish physical frame correctness.
3. Reference acceleration is differentiated from 5 Hz velocity, while the IMU operates at approximately 100 Hz. This limits dynamic comparability and can amplify reference noise.
4. Reference angular rates are derived from 5 Hz attitude samples and are not an independent high-rate angular-rate truth source.
5. The lever arm was recorded but not applied.
6. No automatic axis inversion or sign correction was performed.
7. No ESKF, training, or Real-vs-Synthetic benchmark was executed.
8. The audit's trajectory preview was formed by integrating sampled velocity for visualization; it must not be treated as an authoritative position trajectory.

## 9. Decision

**Current gate: BLOCKED — PHYSICAL FRAME VERIFICATION.**

The dataset schema and temporal checks passed, and the candidate rotation matrices passed numerical SO(3) checks. The accelerometer and gyroscope comparisons still show material discrepancies. Do not use these results as physical validation of the Vyra Navigation Core and do not start the real-data ESKF benchmark yet.

## 10. Next controlled step

PPC Frame Audit v1.3 — Convention Isolation:

- confirm ADIS16505 sensor axis polarity and installed orientation;
- confirm PPC/Applanix roll, pitch, heading definitions and rotation direction;
- confirm the physical reference-position origin;
- establish a valid static/gravity check;
- characterize the limitations of 5 Hz differentiated reference signals;
- only then determine whether a lever-arm correction is applicable.

Corrections must be justified by documented frame definitions and physical tests, not selected solely to maximize correlation.

## 11. Artifact preservation status

This commit preserves the v1.2 summary and machine-readable metrics. The original executed Colab notebook (`.ipynb`), full-resolution PNG plots, and original raw JSON report have not yet been committed. They must be added from the user's Colab runtime/export before claiming that the complete execution bundle is archived.

No GitHub Actions workflow was added.
