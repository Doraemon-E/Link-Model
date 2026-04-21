# Swift Runtime Model Benchmark Summary

- generated_at: `2026-04-20T13:36:27.098Z`
- coreml_compute_unit: `cpuAndNeuralEngine`
- include_mlx: `true`
- total_models: `2`
- success_count: `2`
- failure_count: `0`

## Best

- fastest_load: `Optional(model_bench.BestMetric(runtime: "coreml", variant: "cache-c64", label: "coreml:cache-c64", value: 0.13))`
- fastest_translation: `Optional(model_bench.BestMetric(runtime: "mlx", variant: "hy-mt1.5-1.8b-mlx", label: "mlx:hy-mt1.5-1.8b-mlx", value: 2.341))`
- lowest_load_memory: `Optional(model_bench.BestMetric(runtime: "coreml", variant: "cache-c64", label: "coreml:cache-c64", value: 39.953))`

## Per Model

| runtime | variant | load_seconds | translation_total_seconds | memory_before_load_mb | memory_after_load_mb | memory_delta_load_mb |
|---|---|---:|---:|---:|---:|---:|
| coreml | cache-c64 | 0.13 | 6.42 | 12.938 | 39.953 | 27.016 |
| mlx | hy-mt1.5-1.8b-mlx | 1.659 | 2.341 | 26.891 | 871.922 | 845.031 |
