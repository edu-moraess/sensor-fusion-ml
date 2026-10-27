# Evaluation Protocol

## Rules

1. Metrics are computed **only** when valid ground-truth references exist for the evaluated quantity.
2. Training loss is never reported as a substitute for test-set performance.
3. Hyper-parameters are tuned exclusively on the validation split.
4. Test sequences are never used for model selection.
5. Splits are formed by **sequence / scene**, never by random frame shuffling.
6. Every report records: dataset (or “synthetic”), split, config hash, checkpoint id, units, number of samples.

## Implemented Metrics

| Metric | Quantity | Unit |
|--------|----------|------|
| MAE position | mean absolute error | m |
| RMSE position | root mean square error | m |
| MAE / RMSE velocity | same for velocity | m/s |
| ADE | mean L2 over all future steps | m |
| FDE | L2 at final predicted step | m |

## Synthetic vs Real

- Results obtained on synthetic data are labeled **“software validation only”**.
- They must not be cited as evidence of real-world accuracy.
