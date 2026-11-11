# MPE Detection → Vyra Core → Sensor Fusion PIXEL consumer

## Status

Implementation candidate on branch `feat/mpe-pixel-detection-consumer`.
The consumer is intentionally narrow: native MPE `Detection` only, using
`mpe.image_bbox_xywh.v1`, with explicit Core `PIXEL / IMAGE_2D` semantics.
It does not consume Track or TrackState.

## Optional integration dependencies

The Sensor Fusion package remains independently usable. To use this adapter,
install the local MPE and Vyra Core packages in the same environment. The
adapter imports them lazily and raises `VyraEnvelopeError` if they are absent.

## Required input

- A real Vyra Core `ArtifactEnvelope` with `artifact_type="detection"` and
  `schema_name="mpe.detection"`.
- A native MPE `Detection` payload.
- A valid `mpe.image_bbox_xywh.v1` geometry mapping, with positive image
  dimensions, explicit image reference, original-image coordinate space, and
  an in-bounds XYWH box.
- A Core `SpatialReference` with `coordinate_frame="PIXEL"`,
  `coordinate_system="IMAGE_2D"`, `axis_order=("u", "v")`,
  `units=("px", "px")`, no CRS, and origin
  `top-left:{image_reference}`.
- A timezone-aware capture timestamp and matching source ID, sequence, and
  clock domain in the Core temporal reference.

If preprocessing is applied, transform ID and provenance must agree between
the geometry profile and Core spatial reference.

## Mapping rules

- Output modality is `DETECTION`; frame is `PIXEL`; sensor ID is the MPE
  source ID.
- The original XYWH geometry and class label are preserved in `Observation.data`.
- Capture datetime is represented as Unix seconds in Sensor Fusion while the
  original ISO timestamp, clock domain, source ID, and sequence remain in
  metadata.
- Detector score remains `detector_score`; categorical MPE quality remains
  `mpe_quality`; numeric Observation quality is `None`.
- Evidence mapping: MEASURED → `is_estimate=False`; INFERRED, DERIVED,
  PREDICTED, HYPOTHESIS → `True`. EXTERNAL is rejected pending an explicit
  source-specific policy.
- Artifact identity, native detection identity, producer, repository/commit
  when supplied, and Core provenance are retained.

## Deliberate exclusions

No pixel-to-metric/world projection, calibration, depth inference, clipping,
tracking, TrackState consumption, covariance synthesis, or cross-sensor
synchronization is performed. A timezone-aware timestamp alone does not
establish synchronized clocks.

## Validation

Integration tests use native MPE Detection objects and the actual Core
MPEProducerAdapter and ArtifactEnvelope. They require both optional packages
to be installed in the test environment. Test execution is tracked separately
from source publication; no result is implied by the presence of these tests.
