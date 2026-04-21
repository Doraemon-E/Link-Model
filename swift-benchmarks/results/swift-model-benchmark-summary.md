# Swift Runtime Model Benchmark Summary

- generated_at: `2026-04-21T01:03:36.409Z`
- coreml_compute_unit: `cpuAndNeuralEngine`
- include_mlx: `false`
- total_models: `1`
- success_count: `1`
- failure_count: `0`

## Best

- fastest_load: `Optional(model_bench.BestMetric(runtime: "coreml", variant: "cache-c64", label: "coreml:cache-c64", value: 0.136))`
- fastest_translation: `Optional(model_bench.BestMetric(runtime: "coreml", variant: "cache-c64", label: "coreml:cache-c64", value: 10.201))`
- lowest_load_memory: `Optional(model_bench.BestMetric(runtime: "coreml", variant: "cache-c64", label: "coreml:cache-c64", value: 40.547))`

## Per Model

| runtime | variant | load_seconds | translation_total_seconds | memory_before_load_mb | memory_after_load_mb | memory_delta_load_mb |
|---|---|---:|---:|---:|---:|---:|
| coreml | cache-c64 | 0.136 | 10.201 | 12.984 | 40.547 | 27.563 |
