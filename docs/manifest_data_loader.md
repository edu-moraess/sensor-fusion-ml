# Manifest-based local data loader (v1)

The first loader accepts an explicit local JSON manifest and **never downloads data**.
It does not install packages, mount Drive, start training, or silently remove records.

## Supported payload formats

- `npy`: NumPy array loaded with `allow_pickle=False`.
- `json`: JSON payload.
- `csv`: numeric CSV payload.
- `path`: returns a local path reference without decoding image/video bytes; a modality-specific decoder can be added later.

Every payload path is resolved under the explicit dataset root (or the manifest's parent by default). Paths that escape the root are rejected.

## Manifest schema

```json
{
  "name": "local-sequence-example",
  "license": "license identifier or internal authorization reference",
  "provenance": "where the data came from and how it was acquired",
  "time_unit": "seconds",
  "sequences": [
    {
      "sequence_id": "drive-001",
      "split": "train",
      "frames": [
        {
          "frame_id": "frame-0001",
          "timestamp": 0.0,
          "observations": [
            {
              "modality": "gnss",
              "timestamp": 0.0,
              "time_unit": "seconds",
              "path": "payload/gnss_0001.npy",
              "format": "npy",
              "frame": "world",
              "sensor_id": "gnss-main",
              "quality": 1.0,
              "is_valid": true,
              "is_estimate": false
            }
          ]
        }
      ]
    }
  ]
}
```

Each frame currently supports at most one observation per modality. Sensor-specific calibration and transforms are not inferred by the loader. Timestamp alignment is validated against the supplied DataContract; the loader never assumes perfect synchronization.

## Colab / Python usage

```python
from sensor_fusion_ml.data_engine.contracts import DataContract, SensorModality
from sensor_fusion_ml.data_engine.loader import load_manifest, validate_dataset, validation_passed

dataset = load_manifest("/content/my_dataset/manifest.json")
contract = DataContract(
    name="my-local-dataset",
    required_modalities=[SensorModality.GNSS, SensorModality.IMU],
    min_frames=2,
    max_time_gap_s=0.5,
)
reports = validate_dataset(dataset, contract)
for sequence_id, report in reports.items():
    print(sequence_id, report)
if not validation_passed(reports):
    raise RuntimeError("Data validation did not pass; do not train.")
```

Use only datasets whose license and provenance have been reviewed for the intended research or commercial use. Passing the software validator does not establish scientific validity, sensor calibration, or license compliance.
