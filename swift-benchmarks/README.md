# Swift Model Benchmarks

This folder contains a pure Swift benchmark CLI for comparing:

- CoreML int8 variants under `models/translation/converted/coreml-int8`
- MLX local model under `models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx`

The benchmark records per model:

- RSS memory before load
- RSS memory after load
- RSS memory delta during load
- RSS memory after generation
- load time
- generation time
- total translation time

## Why `xcodebuild` (not `swift build`)

`mlx-swift` requires Metal shader libraries (`metallib`).
Per the official `mlx-swift` README, command-line SwiftPM cannot build Metal shaders, so MLX runtime may fail if built with `swift build` only.

Use `xcodebuild` to build this benchmark binary.

## Run

From `link-model/swift-benchmarks`:

```bash
./run_benchmark.sh \
  --variants auto \
  --include-mlx \
  --compute-unit cpuAndNeuralEngine \
  --max-new-tokens 64 \
  --context-length 256
```

Results are written to:

- `swift-benchmarks/results/swift-model-benchmark-results.json`
- `swift-benchmarks/results/swift-model-benchmark-summary.json`
- `swift-benchmarks/results/swift-model-benchmark-summary.md`

## Local-only MLX load

MLX is loaded from local path with:

- `ModelConfiguration(directory: ...)`
- `LLMModelFactory.shared.loadContainer(configuration: ...)`

No model ID is used for remote Hub fetching in this path.
