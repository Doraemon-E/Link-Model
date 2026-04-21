# Runtime Model Benchmark Summary

- generated_at: `2026-04-20T13:01:19.906782+00:00`
- coreml_compute_unit: `cpuAndNeuralEngine`
- include_mlx: `True`
- total_models: `8`
- success_count: `8`
- failure_count: `0`

## Best

- fastest_load: `{'runtime': 'coreml', 'variant': 'nocache', 'label': 'coreml:nocache', 'value': 0.137}`
- fastest_translation: `{'runtime': 'mlx', 'variant': 'hy-mt1.5-1.8b-mlx', 'label': 'mlx:hy-mt1.5-1.8b-mlx', 'value': 2.046}`
- lowest_load_memory: `{'runtime': 'coreml', 'variant': 'cache-c128', 'label': 'coreml:cache-c128', 'value': 410.984}`

## Per Model

| runtime | variant | load_seconds | translation_total_seconds | memory_before_load_mb | memory_after_load_mb | memory_delta_load_mb |
|---|---|---:|---:|---:|---:|---:|
| coreml | cache | 0.15 | 8.522 | 414.344 | 437.094 | 22.75 |
| coreml | cache-c128 | 0.151 | 8.937 | 388.219 | 410.984 | 22.766 |
| coreml | cache-c256 | 0.157 | 8.812 | 418.578 | 441.594 | 23.016 |
| coreml | cache-c64 | 0.152 | 8.461 | 416.469 | 440.438 | 23.969 |
| coreml | cache-c96 | 0.151 | 8.593 | 412.812 | 434.281 | 21.469 |
| coreml | cache-opt | 0.15 | 8.479 | 414.453 | 437.562 | 23.109 |
| coreml | nocache | 0.137 | 10.338 | 416.625 | 440.375 | 23.75 |
| mlx | hy-mt1.5-1.8b-mlx | 1.071 | 2.046 | 293.547 | 432.562 | 139.016 |
