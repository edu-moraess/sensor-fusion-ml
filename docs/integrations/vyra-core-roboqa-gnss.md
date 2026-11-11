# Vyra Core — RoboQA GNSS consumer bridge

## Scope and ownership

This optional integration consumes a Vyra Core envelope carrying a native
RoboQA GPS Measurement and creates a Sensor Fusion ML
`Observation(modality=GNSS)`.

- RoboQA owns acquisition, native measurement and source covariance.
- Vyra Core owns envelope identity, provenance, temporal/spatial declarations,
  compatibility resolution and in-process routing.
- Sensor Fusion owns the consumer mapping, typed Observation and estimator
  update policy.

The Sensor Fusion package remains independently usable. Neither RoboQA nor
Vyra Core is imported as a runtime dependency by the base package.

## Core adapter protocol and compatibility declaration

`RoboQAGPSConsumerAdapter` implements the Core structural consumer protocol:
`name`, `version`, and `from_envelope(envelope)`.

Canonical values for the composition root:

- producer module: `vyra.roboqa`
- producer identity: `roboqa-lab`
- source schema: `roboqa.measurement`
- source artifact: `gps_measurement`
- source schema version: `0.1.0`
- consumer module: `vyra.sensor_fusion`
- adapter: `roboqa-gps-consumer@0.1.0`
- consumer capability: `sensor_fusion.observation:gnss_observation@0.1.0`

The host application must register the producer and consumer declarations and
an explicit `CompatibilityDeclaration` matching these values. The consumer
adapter must be registered under `vyra.sensor_fusion`. No implicit schema
conversion or fallback is supported.

Example composition:

```python
from vyra_core import (
    ArtifactCapability,
    IntegrationRouter,
    ModuleDeclaration,
    ModuleRegistry,
)
from vyra_core.validation import CompatibilityDeclaration
from sensor_fusion_ml.integrations import RoboQAGPSConsumerAdapter

producer = ModuleDeclaration(
    module_id="vyra.roboqa",
    module_version="0.1.0",
    produces=(ArtifactCapability(
        "roboqa.measurement", "gps_measurement", ("0.1.0",)
    ),),
    producer_ids=("roboqa-lab",),
)
consumer = ModuleDeclaration(
    module_id="vyra.sensor_fusion",
    module_version="0.1.0",
    consumes=(ArtifactCapability(
        "sensor_fusion.observation", "gnss_observation", ("0.1.0",)
    ),),
)
compatibility = CompatibilityDeclaration(
    producer_module_id="vyra.roboqa",
    producer_id="roboqa-lab",
    schema_name="roboqa.measurement",
    artifact_type="gps_measurement",
    supported_schema_versions=("0.1.0",),
    consumer="vyra.sensor_fusion",
    consumer_schema_name="sensor_fusion.observation",
    consumer_artifact_type="gnss_observation",
    consumer_schema_versions=("0.1.0",),
    adapter_name="roboqa-gps-consumer",
    adapter_version="0.1.0",
)
registry = ModuleRegistry(
    declarations=(producer, consumer),
    compatibility_declarations=(compatibility,),
)
router = IntegrationRouter(registry)
router.register_consumer("vyra.sensor_fusion", RoboQAGPSConsumerAdapter())
```

This example declares and registers the in-process route. It does not claim
that a real RoboQA artifact has been published or delivered.

## Accepted producer spatial semantics

The current consumer accepts only:

- `coordinate_frame="LOCAL_CARTESIAN"`
- `axis_order=("x", "y")`
- `units=("m", "m")`
- explicit `coordinate_system`, `crs`, and `origin`

The RoboQA KITTI loader uses an equirectangular local approximation with the
first OXTS sample as origin. That is not silently renamed ENU. The consumer
preserves the declared local Cartesian frame and performs no coordinate
transformation.

Geographic latitude/longitude, implicit UTM assumptions, and undeclared
transforms are rejected.

## Time semantics

The envelope event time must equal native `Measurement.t`, be expressed in
seconds, and declare `time_reference` and `clock_domain`.

The current KITTI OXTS loader approximates timestamps at 10 Hz. Those values
must not be labelled UTC or synchronized wall-clock time without independent
evidence. The RoboQA fixture's `not_synchronized` status is preserved; it is
not upgraded by this adapter.

## Covariance semantics

The native 2x2 `R` is validated and represented as first-class
`Observation.uncertainty`:

- status: `provided`
- variable order: `("x", "y")`
- units: `("m", "m")`
- frame: `LOCAL_CARTESIAN`
- provenance: source Vyra artifact ID

The covariance remains PSD-compatible and is not modified. The legacy
`metadata["measurement_covariance"]` JSON list remains for transition
compatibility; typed uncertainty is authoritative.

The ESKF measurement update requires positive-definite covariance. A singular
PSD covariance is rejected at the ESKF bridge boundary; no diagonal loading,
epsilon injection, or other regularization is performed.

## ESKF safety gate

`apply_enu_gnss_observation` is a separate, fail-closed operation. It accepts
only an Observation that already declares:

- `CoordinateFrame.ENU`, East/North axis order and metre units;
- ENU covariance with variable order `("east", "north")`;
- explicit CRS, origin, transform ID and transform provenance;
- a measurement reference matching the ESKF state's reference;
- temporal synchronization explicitly marked `synchronized`, with the Observation timestamp source matching the declared clock domain;
- finite synchronization uncertainty within a caller-supplied tolerance;
- measurement timestamp within a caller-supplied offset of an explicit ESKF state time;
- a positive-definite 2x2 covariance.

It performs no transform or clock synchronization. The caller must supply
`state_time_s`, `max_measurement_time_offset_s`, and
`max_synchronization_uncertainty_s`; these tolerances are application policy,
not defaults inferred by the adapter. The current RoboQA consumer output is
`LOCAL_CARTESIAN` and the KITTI time basis is not synchronized, so it is
expected to be rejected by this ESKF gate until a separately validated spatial
transform and time-alignment path exists.

## API

```python
from sensor_fusion_ml.integrations import (
    RoboQAGPSConsumerAdapter,
    roboqa_gps_envelope_to_observation,
)

consumer = RoboQAGPSConsumerAdapter()
observation = consumer.from_envelope(vyra_envelope)
```

The adapter can deliver a typed local Cartesian GNSS Observation through the
Core router. That is not yet an ESKF update or a scientific navigation result.

## Validation status

Unit tests cover the adapter's structural protocol, envelope-to-Observation
conversion, and fail-closed ESKF preconditions. A three-repository Colab run
must still verify the actual Core registry/router delivery using the exact
commits and dependencies. No real dataset, training run, coordinate
transformation, clock synchronization, or scientific experiment is implied.
