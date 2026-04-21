// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ModelBench",
    platforms: [
        .macOS("15.0"),
    ],
    products: [
        .executable(name: "model-bench", targets: ["ModelBench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.3")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "2.30.3")),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "1.1.0")
        ),
    ],
    targets: [
        .executableTarget(
            name: "ModelBench",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/ModelBench"
        ),
    ]
)
