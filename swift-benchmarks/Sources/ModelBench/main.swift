import CoreML
import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon

enum BenchError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case let .message(text):
            return text
        }
    }
}

struct CLIConfig {
    var linkModelRoot: URL
    var coremlRootDir: URL
    var artifactStem: String
    var variantsRaw: String
    var includeMLX: Bool
    var mlxModelDir: URL
    var pythonExecutable: String
    var computeUnitName: String
    var sourceText: String
    var targetLanguage: String
    var systemPrompt: String
    var maxNewTokens: Int
    var requestedContextLength: Int
    var continueOnError: Bool
    var resultsDir: URL
    var resultsJSONName: String
    var summaryJSONName: String
    var summaryMDName: String

    static func defaults(from cwd: URL) -> CLIConfig {
        let root = detectLinkModelRoot(from: cwd)
        return CLIConfig(
            linkModelRoot: root,
            coremlRootDir: root.appendingPathComponent("models/translation/converted/coreml-int8", isDirectory: true),
            artifactStem: "hy-mt1.5-1.8b-coreml-int8",
            variantsRaw: "auto",
            includeMLX: true,
            mlxModelDir: root.appendingPathComponent("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx", isDirectory: true),
            pythonExecutable: defaultPythonExecutable(root: root),
            computeUnitName: "cpuAndNeuralEngine",
            sourceText: "今天下午三点半在5A会议室开会。",
            targetLanguage: "English",
            systemPrompt: "You are a translation engine.",
            maxNewTokens: 64,
            requestedContextLength: 256,
            continueOnError: true,
            resultsDir: root.appendingPathComponent("swift-benchmarks/results", isDirectory: true),
            resultsJSONName: "swift-model-benchmark-results.json",
            summaryJSONName: "swift-model-benchmark-summary.json",
            summaryMDName: "swift-model-benchmark-summary.md"
        )
    }
}

struct ManifestInfo {
    let path: URL
    let raw: [String: Any]
    let contextLength: Int?
    let inputName: String?
    let outputName: String?
    let modelPath: String?
}

struct InferenceMetrics: Codable {
    var variant: String?
    var computeUnit: String?
    var compiledMaterialized: Bool?
    var statefulRuntime: Bool?
    var stateError: String?
    var loadSeconds: Double?
    var prefillSeconds: Double?
    var firstTokenLatencySeconds: Double?
    var generateSeconds: Double?
    var translationTotalSeconds: Double?
    var endToEndSeconds: Double?
    var promptTokens: Int?
    var outputTokens: Int?
    var memoryRSSBeforeLoadMB: Double?
    var memoryRSSAfterLoadMB: Double?
    var memoryRSSAfterGenerateMB: Double?
    var memoryRSSDeltaLoadMB: Double?
    var memoryRSSBeforeLoadBytes: Int64?
    var memoryRSSAfterLoadBytes: Int64?
    var memoryRSSAfterGenerateBytes: Int64?
    var memoryRSSDeltaLoadBytes: Int64?
}

struct BenchmarkRow: Codable {
    var status: String
    var runtime: String
    var variant: String
    var coremlDir: String?
    var mlxModelDir: String?
    var modelPath: String?
    var inference: InferenceMetrics?
    var error: String?
    var effectiveContextLength: Int?
}

struct BestMetric: Codable {
    let runtime: String
    let variant: String
    let label: String
    let value: Double
}

struct SummaryModelRow: Codable {
    let runtime: String
    let variant: String
    let label: String
    let coremlDir: String?
    let mlxModelDir: String?
    let modelPath: String?
    let statefulRuntime: Bool?
    let loadSeconds: Double?
    let translationTotalSeconds: Double?
    let generateSeconds: Double?
    let memoryRSSBeforeLoadMB: Double?
    let memoryRSSAfterLoadMB: Double?
    let memoryRSSDeltaLoadMB: Double?
    let memoryRSSAfterGenerateMB: Double?
    let promptTokens: Int?
    let outputTokens: Int?
}

struct RankedMetricEntry: Codable {
    let rank: Int
    let runtime: String
    let variant: String
    let label: String
    let metric: String
    let value: Double
}

struct SummaryReport: Codable {
    let status: String
    let generatedAt: String
    let coremlComputeUnit: String
    let includeMLX: Bool
    let coremlRootDir: String
    let mlxModelDir: String
    let artifactStem: String
    let coremlVariants: [String]
    let totalModels: Int
    let successCount: Int
    let failureCount: Int
    let fastestLoad: BestMetric?
    let fastestTranslation: BestMetric?
    let lowestLoadMemory: BestMetric?
    let modelRows: [SummaryModelRow]
    let failures: [BenchmarkRow]
    let rankings: [String: [RankedMetricEntry]]
}

struct FullResults: Codable {
    struct ConfigSnapshot: Codable {
        let coremlComputeUnit: String
        let sourceText: String
        let targetLanguage: String
        let systemPrompt: String
        let maxNewTokens: Int
        let requestedContextLength: Int
        let coremlRootDir: String
        let artifactStem: String
        let coremlVariants: [String]
        let includeMLX: Bool
        let mlxModelDir: String
        let pythonExecutable: String
    }

    let status: String
    let generatedAt: String
    let config: ConfigSnapshot
    let rows: [BenchmarkRow]
    let summary: SummaryReport
}

struct ProcessOutput {
    let status: Int32
    let stdout: String
    let stderr: String
}

struct TokenizeResponse: Codable {
    let ids: [Int]
    let eos_ids: [Int]
}

struct DecodeResponse: Codable {
    let text: String
}

let fileManager = FileManager.default

func detectLinkModelRoot(from cwd: URL) -> URL {
    var current = cwd
    for _ in 0..<8 {
        let marker = current.appendingPathComponent("covert_to_coreml.py")
        let coremlRoot = current.appendingPathComponent("models/translation/converted/coreml-int8")
        if fileManager.fileExists(atPath: marker.path) || fileManager.fileExists(atPath: coremlRoot.path) {
            return current
        }
        let parent = current.deletingLastPathComponent()
        if parent.path == current.path {
            break
        }
        current = parent
    }
    return cwd
}

func defaultPythonExecutable(root: URL) -> String {
    let preferred = root.appendingPathComponent(".venv/bin/python")
    if fileManager.isExecutableFile(atPath: preferred.path) {
        return preferred.path
    }
    return "python3"
}

func printUsage(defaults: CLIConfig) {
    let text = """
    Usage: model-bench [options]

    Options:
      --link-model-root <path>
      --coreml-root-dir <path>
      --artifact-stem <name>
      --variants <auto|comma-list>
      --include-mlx | --no-include-mlx
      --mlx-model-dir <path>
      --python-exec <path-or-command>
      --compute-unit <cpuAndNeuralEngine|all|cpuAndGPU|cpuOnly>
      --source-text <text>
      --target-language <lang>
      --system-prompt <text>
      --max-new-tokens <int>
      --context-length <int>
      --continue-on-error | --no-continue-on-error
      --results-dir <path>
      --results-json <filename>
      --summary-json <filename>
      --summary-md <filename>
      --help

    Defaults:
      link-model-root: \(defaults.linkModelRoot.path)
      coreml-root-dir: \(defaults.coremlRootDir.path)
      variants: \(defaults.variantsRaw)
      include-mlx: \(defaults.includeMLX)
      mlx-model-dir: \(defaults.mlxModelDir.path)
      python-exec: \(defaults.pythonExecutable)
      compute-unit: \(defaults.computeUnitName)
      max-new-tokens: \(defaults.maxNewTokens)
      context-length: \(defaults.requestedContextLength)
      results-dir: \(defaults.resultsDir.path)
    """
    print(text)
}

func parseCLI() throws -> CLIConfig {
    let cwd = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
    var config = CLIConfig.defaults(from: cwd)

    var args = CommandLine.arguments
    _ = args.removeFirst()

    var index = 0
    while index < args.count {
        let arg = args[index]
        switch arg {
        case "--help", "-h":
            printUsage(defaults: config)
            exit(0)
        case "--link-model-root":
            index += 1
            let root = URL(fileURLWithPath: try takeValue(args, index, flag: arg), isDirectory: true)
            config.linkModelRoot = root
            config.coremlRootDir = root.appendingPathComponent("models/translation/converted/coreml-int8", isDirectory: true)
            config.mlxModelDir = root.appendingPathComponent("models/translation/converted/mlx-int8/hy-mt1.5-1.8b-mlx", isDirectory: true)
            config.resultsDir = root.appendingPathComponent("swift-benchmarks/results", isDirectory: true)
            if config.pythonExecutable == defaultPythonExecutable(root: CLIConfig.defaults(from: cwd).linkModelRoot) {
                config.pythonExecutable = defaultPythonExecutable(root: root)
            }
        case "--coreml-root-dir":
            index += 1
            config.coremlRootDir = URL(fileURLWithPath: try takeValue(args, index, flag: arg), isDirectory: true)
        case "--artifact-stem":
            index += 1
            config.artifactStem = try takeValue(args, index, flag: arg)
        case "--variants":
            index += 1
            config.variantsRaw = try takeValue(args, index, flag: arg)
        case "--include-mlx":
            config.includeMLX = true
        case "--no-include-mlx":
            config.includeMLX = false
        case "--mlx-model-dir":
            index += 1
            config.mlxModelDir = URL(fileURLWithPath: try takeValue(args, index, flag: arg), isDirectory: true)
        case "--python-exec":
            index += 1
            config.pythonExecutable = try takeValue(args, index, flag: arg)
        case "--compute-unit":
            index += 1
            config.computeUnitName = try takeValue(args, index, flag: arg)
        case "--source-text":
            index += 1
            config.sourceText = try takeValue(args, index, flag: arg)
        case "--target-language":
            index += 1
            config.targetLanguage = try takeValue(args, index, flag: arg)
        case "--system-prompt":
            index += 1
            config.systemPrompt = try takeValue(args, index, flag: arg)
        case "--max-new-tokens":
            index += 1
            config.maxNewTokens = Int(try takeValue(args, index, flag: arg)) ?? config.maxNewTokens
        case "--context-length":
            index += 1
            config.requestedContextLength = Int(try takeValue(args, index, flag: arg)) ?? config.requestedContextLength
        case "--continue-on-error":
            config.continueOnError = true
        case "--no-continue-on-error":
            config.continueOnError = false
        case "--results-dir":
            index += 1
            config.resultsDir = URL(fileURLWithPath: try takeValue(args, index, flag: arg), isDirectory: true)
        case "--results-json":
            index += 1
            config.resultsJSONName = try takeValue(args, index, flag: arg)
        case "--summary-json":
            index += 1
            config.summaryJSONName = try takeValue(args, index, flag: arg)
        case "--summary-md":
            index += 1
            config.summaryMDName = try takeValue(args, index, flag: arg)
        default:
            throw BenchError.message("unknown argument: \(arg)")
        }
        index += 1
    }

    if config.maxNewTokens <= 0 {
        throw BenchError.message("--max-new-tokens must be > 0")
    }
    if config.requestedContextLength <= 0 {
        throw BenchError.message("--context-length must be > 0")
    }

    return config
}

func takeValue(_ args: [String], _ index: Int, flag: String) throws -> String {
    if index >= args.count {
        throw BenchError.message("missing value for \(flag)")
    }
    return args[index]
}

func runProcess(
    executable: String,
    arguments: [String],
    currentDirectory: URL? = nil,
    environment: [String: String]? = nil
) throws -> ProcessOutput {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments
    if let currentDirectory {
        process.currentDirectoryURL = currentDirectory
    }

    if let environment {
        process.environment = environment
    }

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    try process.run()
    process.waitUntilExit()

    let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

    return ProcessOutput(
        status: process.terminationStatus,
        stdout: String(decoding: stdoutData, as: UTF8.self),
        stderr: String(decoding: stderrData, as: UTF8.self)
    )
}

func currentRSSBytes() -> Int64? {
    do {
        let output = try runProcess(
            executable: "/bin/ps",
            arguments: ["-o", "rss=", "-p", "\(getpid())"]
        )
        guard output.status == 0 else {
            return nil
        }
        let trimmed = output.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let kb = Int64(trimmed) else {
            return nil
        }
        return kb * 1024
    } catch {
        return nil
    }
}

func bytesToMB(_ bytes: Int64?) -> Double? {
    guard let bytes else {
        return nil
    }
    return round((Double(bytes) / (1024.0 * 1024.0)) * 1000.0) / 1000.0
}

func iso8601Now() -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter.string(from: Date())
}

func buildPrompt(targetLanguage: String, sourceText: String) -> String {
    return "将以下文本翻译为\(targetLanguage)，注意只需要输出翻译后的结果，不要额外解释：\n\n\(sourceText)"
}

func loadManifest(coremlDir: URL) throws -> ManifestInfo {
    let manifestURL = coremlDir.appendingPathComponent("translation-manifest.json")
    let data = try Data(contentsOf: manifestURL)
    let rawObj = try JSONSerialization.jsonObject(with: data, options: [])
    guard let raw = rawObj as? [String: Any] else {
        throw BenchError.message("manifest is not a JSON object: \(manifestURL.path)")
    }

    var contextLength: Int?
    if let value = raw["contextLength"] as? Int {
        contextLength = value
    }

    var inputName: String?
    var outputName: String?
    var modelPath: String?

    if let coreml = raw["coreml"] as? [String: Any] {
        inputName = coreml["inputName"] as? String
        outputName = coreml["outputName"] as? String
        modelPath = coreml["path"] as? String
    }

    return ManifestInfo(
        path: manifestURL,
        raw: raw,
        contextLength: contextLength,
        inputName: inputName,
        outputName: outputName,
        modelPath: modelPath
    )
}

func resolveModelPath(coremlDir: URL, manifestModelPath: String?) throws -> URL {
    if let manifestModelPath, !manifestModelPath.isEmpty {
        let candidate: URL
        if manifestModelPath.hasPrefix("/") {
            candidate = URL(fileURLWithPath: manifestModelPath, isDirectory: true)
        } else {
            candidate = coremlDir.appendingPathComponent(manifestModelPath)
        }
        if fileManager.fileExists(atPath: candidate.path) {
            return candidate
        }
    }

    let fallbackCompiled = coremlDir.appendingPathComponent("hy_mt_w8_from_torch.mlmodelc", isDirectory: true)
    if fileManager.fileExists(atPath: fallbackCompiled.path) {
        return fallbackCompiled
    }

    let fallbackPackage = coremlDir.appendingPathComponent("hy_mt_w8_from_torch.mlpackage", isDirectory: true)
    if fileManager.fileExists(atPath: fallbackPackage.path) {
        return fallbackPackage
    }

    throw BenchError.message("missing model artifact under \(coremlDir.path)")
}

func updateManifestCoreMLTarget(
    manifestURL: URL,
    modelPath: URL,
    coremlDir: URL
) throws {
    let data = try Data(contentsOf: manifestURL)
    let rawObj = try JSONSerialization.jsonObject(with: data, options: [])
    guard var payload = rawObj as? [String: Any] else {
        throw BenchError.message("manifest payload invalid: \(manifestURL.path)")
    }

    var coreml = payload["coreml"] as? [String: Any] ?? [:]

    let relativePath: String
    if modelPath.path.hasPrefix(coremlDir.path + "/") {
        relativePath = String(modelPath.path.dropFirst(coremlDir.path.count + 1))
    } else {
        relativePath = modelPath.path
    }

    coreml["path"] = relativePath
    coreml["kind"] = modelPath.pathExtension == "mlmodelc" ? "mlmodelc" : "mlpackage"
    payload["coreml"] = coreml

    let updated = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    if var text = String(data: updated, encoding: .utf8) {
        text.append("\n")
        try text.write(to: manifestURL, atomically: true, encoding: .utf8)
    } else {
        try updated.write(to: manifestURL)
    }
}

func removePathIfExists(_ url: URL) throws {
    if fileManager.fileExists(atPath: url.path) {
        try fileManager.removeItem(at: url)
    }
}

func materializeCompiledModelIfNeeded(coremlDir: URL, modelPath: URL, manifestURL: URL) throws -> (URL, Bool) {
    if modelPath.pathExtension != "mlpackage" {
        return (modelPath, false)
    }

    let compiledPath = modelPath.deletingPathExtension().appendingPathExtension("mlmodelc")
    var created = false
    if !fileManager.fileExists(atPath: compiledPath.path) {
        let tempCompiled = try MLModel.compileModel(at: modelPath)
        try removePathIfExists(compiledPath)
        try fileManager.copyItem(at: tempCompiled, to: compiledPath)
        created = true
    }

    try updateManifestCoreMLTarget(manifestURL: manifestURL, modelPath: compiledPath, coremlDir: coremlDir)
    return (compiledPath, created)
}

func resolveComputeUnit(_ name: String) throws -> MLComputeUnits {
    switch name {
    case "cpuAndNeuralEngine":
        return .cpuAndNeuralEngine
    case "all":
        return .all
    case "cpuAndGPU":
        return .cpuAndGPU
    case "cpuOnly":
        return .cpuOnly
    default:
        throw BenchError.message("unsupported compute unit: \(name)")
    }
}

func resolveOutputName(model: MLModel, manifestOutputName: String?) throws -> String {
    let outputs = model.modelDescription.outputDescriptionsByName
    if let manifestOutputName, outputs[manifestOutputName] != nil {
        return manifestOutputName
    }
    if outputs["logits"] != nil {
        return "logits"
    }
    if let first = outputs.keys.sorted().first {
        return first
    }
    throw BenchError.message("model has no outputs")
}

func resolveInputMaxTokens(model: MLModel, inputName: String) -> Int? {
    guard let feature = model.modelDescription.inputDescriptionsByName[inputName] else {
        return nil
    }
    guard let constraint = feature.multiArrayConstraint else {
        return nil
    }

    let shape = constraint.shape.map { $0.intValue }
    if shape.count >= 2, shape[1] > 0 {
        return shape[1]
    }
    if shape.count == 1, shape[0] > 0 {
        return shape[0]
    }
    return nil
}

func makeInputProvider(inputName: String, tokens: [Int]) throws -> MLDictionaryFeatureProvider {
    if tokens.isEmpty {
        throw BenchError.message("empty token input")
    }
    let seqLen = tokens.count
    let array = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
    let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: seqLen)
    let stride = array.strides.count > 1 ? array.strides[1].intValue : 1
    for (idx, token) in tokens.enumerated() {
        ptr[idx * stride] = Int32(token)
    }
    return try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: array)])
}

func prediction(
    model: MLModel,
    provider: MLFeatureProvider,
    state: MLState?
) throws -> MLFeatureProvider {
    if let state {
        return try model.prediction(from: provider, using: state)
    }
    return try model.prediction(from: provider)
}

func floatValue(array: MLMultiArray, linearIndex: Int) throws -> Float {
    switch array.dataType {
    case .float16:
        let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
        return Float(Float16(bitPattern: ptr[linearIndex]))
    case .float32:
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        return ptr[linearIndex]
    case .double:
        let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
        return Float(ptr[linearIndex])
    case .int32:
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        return Float(ptr[linearIndex])
    default:
        throw BenchError.message("unsupported logits dtype: \(array.dataType.rawValue)")
    }
}

func argmaxNextToken(from logits: MLMultiArray) throws -> Int {
    let shape = logits.shape.map { $0.intValue }
    let strides = logits.strides.map { $0.intValue }

    var baseOffset = 0
    var vocab = 0
    var vocabStride = 1

    if shape.count == 3 {
        guard shape[1] > 0, shape[2] > 0 else {
            throw BenchError.message("invalid logits shape: \(shape)")
        }
        baseOffset = (shape[1] - 1) * strides[1]
        vocab = shape[2]
        vocabStride = strides[2]
    } else if shape.count == 2 {
        guard shape[0] > 0, shape[1] > 0 else {
            throw BenchError.message("invalid logits shape: \(shape)")
        }
        baseOffset = (shape[0] - 1) * strides[0]
        vocab = shape[1]
        vocabStride = strides[1]
    } else if shape.count == 1 {
        guard shape[0] > 0 else {
            throw BenchError.message("invalid logits shape: \(shape)")
        }
        baseOffset = 0
        vocab = shape[0]
        vocabStride = strides[0]
    } else {
        throw BenchError.message("unexpected logits shape rank: \(shape)")
    }

    var bestToken = 0
    var bestScore = -Float.infinity

    for token in 0..<vocab {
        let offset = baseOffset + token * vocabStride
        let score = try floatValue(array: logits, linearIndex: offset)
        if score > bestScore {
            bestScore = score
            bestToken = token
        }
    }

    return bestToken
}

func tokenizeWithPython(
    python: String,
    tokenizerDir: URL,
    messages: [[String: String]]
) throws -> TokenizeResponse {
    let script = #"""
import json
import sys
from transformers import AutoTokenizer


def _load_eos_ids(tokenizer_dir, tokenizer):
    eos_ids = set()
    generation_config_path = tokenizer_dir + "/generation_config.json"
    try:
        with open(generation_config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        eos_raw = payload.get("eos_token_id")
        if isinstance(eos_raw, int):
            eos_ids.add(eos_raw)
        elif isinstance(eos_raw, list):
            for v in eos_raw:
                if isinstance(v, int):
                    eos_ids.add(v)
    except Exception:
        pass

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)

    return sorted(eos_ids)


tokenizer_dir = sys.argv[1]
messages_json = sys.argv[2]
messages = json.loads(messages_json)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
)
eos_ids = _load_eos_ids(tokenizer_dir, tokenizer)

print(json.dumps({"ids": ids, "eos_ids": eos_ids}, ensure_ascii=False))
"""#

    let messagesData = try JSONSerialization.data(withJSONObject: messages, options: [])
    guard let messagesString = String(data: messagesData, encoding: .utf8) else {
        throw BenchError.message("failed to encode messages json")
    }

    let output = try runProcess(
        executable: python,
        arguments: ["-c", script, tokenizerDir.path, messagesString]
    )
    if output.status != 0 {
        throw BenchError.message("python tokenize failed: \(output.stderr)")
    }
    guard let data = output.stdout.data(using: .utf8) else {
        throw BenchError.message("tokenize stdout is not utf-8")
    }
    return try JSONDecoder().decode(TokenizeResponse.self, from: data)
}

func decodeWithPython(
    python: String,
    tokenizerDir: URL,
    tokenIDs: [Int]
) -> String? {
    if tokenIDs.isEmpty {
        return ""
    }

    let script = #"""
import json
import sys
from transformers import AutoTokenizer

tokenizer_dir = sys.argv[1]
ids = json.loads(sys.argv[2])
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
text = tokenizer.decode(ids, skip_special_tokens=True)
print(json.dumps({"text": text}, ensure_ascii=False))
"""#

    do {
        let tokenData = try JSONSerialization.data(withJSONObject: tokenIDs, options: [])
        guard let tokenString = String(data: tokenData, encoding: .utf8) else {
            return nil
        }

        let output = try runProcess(
            executable: python,
            arguments: ["-c", script, tokenizerDir.path, tokenString]
        )
        if output.status != 0 {
            return nil
        }
        guard let data = output.stdout.data(using: .utf8) else {
            return nil
        }
        let decoded = try JSONDecoder().decode(DecodeResponse.self, from: data)
        return decoded.text
    } catch {
        return nil
    }
}

func runCoreMLBenchmark(
    variant: String,
    coremlDir: URL,
    config: CLIConfig,
    tokenCache: inout [String: TokenizeResponse]
) throws -> BenchmarkRow {
    let startEndToEnd = Date()

    let manifest = try loadManifest(coremlDir: coremlDir)
    var modelPath = try resolveModelPath(coremlDir: coremlDir, manifestModelPath: manifest.modelPath)
    let (materializedPath, compiledMaterialized) = try materializeCompiledModelIfNeeded(
        coremlDir: coremlDir,
        modelPath: modelPath,
        manifestURL: manifest.path
    )
    modelPath = materializedPath

    let effectiveContext = min(config.requestedContextLength, manifest.contextLength ?? config.requestedContextLength)
    let prompt = buildPrompt(targetLanguage: config.targetLanguage, sourceText: config.sourceText)
    let messages = [
        ["role": "system", "content": config.systemPrompt],
        ["role": "user", "content": prompt],
    ]

    let tokenCacheKey = coremlDir.path + "|" + prompt + "|" + config.systemPrompt
    let tokenized: TokenizeResponse
    if let cached = tokenCache[tokenCacheKey] {
        tokenized = cached
    } else {
        let computed = try tokenizeWithPython(
            python: config.pythonExecutable,
            tokenizerDir: coremlDir,
            messages: messages
        )
        tokenCache[tokenCacheKey] = computed
        tokenized = computed
    }

    var promptIDs = tokenized.ids
    if promptIDs.count > effectiveContext {
        promptIDs = Array(promptIDs.suffix(effectiveContext))
    }
    if promptIDs.isEmpty {
        throw BenchError.message("prompt ids is empty for variant=\(variant)")
    }

    let computeUnit = try resolveComputeUnit(config.computeUnitName)

    let translationStart = Date()
    let memoryBeforeLoad = currentRSSBytes()

    let loadStart = Date()
    let modelConfig = MLModelConfiguration()
    modelConfig.computeUnits = computeUnit
    let model = try MLModel(contentsOf: modelPath, configuration: modelConfig)
    let loadSeconds = Date().timeIntervalSince(loadStart)

    let outputName = try resolveOutputName(model: model, manifestOutputName: manifest.outputName)
    let inputName = manifest.inputName ?? "input_ids"

    let hasStates = !model.modelDescription.stateDescriptionsByName.isEmpty
    let state = hasStates ? model.makeState() : nil
    let statefulRuntime = state != nil

    var maxQueryTokens = resolveInputMaxTokens(model: model, inputName: inputName)
    if statefulRuntime, maxQueryTokens == nil {
        maxQueryTokens = 1
    }

    let memoryAfterLoad = currentRSSBytes()

    var prefillSeconds = 0.0
    var currentInputTokens: [Int]
    var tokenHistory = promptIDs

    if statefulRuntime, maxQueryTokens == 1, promptIDs.count > 1 {
        let prefillStart = Date()
        for token in promptIDs.dropLast() {
            let provider = try makeInputProvider(inputName: inputName, tokens: [token])
            _ = try prediction(model: model, provider: provider, state: state)
        }
        prefillSeconds = Date().timeIntervalSince(prefillStart)
        currentInputTokens = [promptIDs.last!]
    } else {
        if let maxQueryTokens, tokenHistory.count > maxQueryTokens {
            tokenHistory = Array(tokenHistory.suffix(maxQueryTokens))
        }
        currentInputTokens = tokenHistory
    }

    let eosSet = Set(tokenized.eos_ids)
    var generatedIDs: [Int] = []
    var stepTimes: [Double] = []

    for _ in 0..<config.maxNewTokens {
        let stepStart = Date()
        let provider = try makeInputProvider(inputName: inputName, tokens: currentInputTokens)
        let output = try prediction(model: model, provider: provider, state: state)
        let stepElapsed = Date().timeIntervalSince(stepStart)
        stepTimes.append(stepElapsed)

        guard let featureValue = output.featureValue(for: outputName)?.multiArrayValue else {
            throw BenchError.message("output missing \(outputName), available=\(output.featureNames)")
        }

        let nextToken = try argmaxNextToken(from: featureValue)
        if eosSet.contains(nextToken) {
            break
        }

        generatedIDs.append(nextToken)
        if statefulRuntime {
            currentInputTokens = [nextToken]
        } else {
            tokenHistory.append(nextToken)
            if let maxQueryTokens, tokenHistory.count > maxQueryTokens {
                tokenHistory = Array(tokenHistory.suffix(maxQueryTokens))
            }
            currentInputTokens = tokenHistory
        }
    }

    let memoryAfterGenerate = currentRSSBytes()
    let translationTotalSeconds = Date().timeIntervalSince(translationStart)
    let endToEndSeconds = Date().timeIntervalSince(startEndToEnd)

    let generatedText = decodeWithPython(
        python: config.pythonExecutable,
        tokenizerDir: coremlDir,
        tokenIDs: generatedIDs
    )?.trimmingCharacters(in: .whitespacesAndNewlines)

    if generatedIDs.isEmpty {
        throw BenchError.message("generated zero tokens for variant=\(variant) text=\(generatedText ?? "")")
    }

    let deltaLoadBytes: Int64?
    if let before = memoryBeforeLoad, let after = memoryAfterLoad {
        deltaLoadBytes = after - before
    } else {
        deltaLoadBytes = nil
    }

    let inference = InferenceMetrics(
        variant: variant,
        computeUnit: config.computeUnitName,
        compiledMaterialized: compiledMaterialized,
        statefulRuntime: statefulRuntime,
        stateError: statefulRuntime ? nil : (hasStates ? "makeState returned nil" : nil),
        loadSeconds: round(loadSeconds * 1000.0) / 1000.0,
        prefillSeconds: round(prefillSeconds * 1000.0) / 1000.0,
        firstTokenLatencySeconds: stepTimes.isEmpty ? nil : round(stepTimes[0] * 1000.0) / 1000.0,
        generateSeconds: round(stepTimes.reduce(0, +) * 1000.0) / 1000.0,
        translationTotalSeconds: round(translationTotalSeconds * 1000.0) / 1000.0,
        endToEndSeconds: round(endToEndSeconds * 1000.0) / 1000.0,
        promptTokens: promptIDs.count,
        outputTokens: generatedIDs.count,
        memoryRSSBeforeLoadMB: bytesToMB(memoryBeforeLoad),
        memoryRSSAfterLoadMB: bytesToMB(memoryAfterLoad),
        memoryRSSAfterGenerateMB: bytesToMB(memoryAfterGenerate),
        memoryRSSDeltaLoadMB: bytesToMB(deltaLoadBytes),
        memoryRSSBeforeLoadBytes: memoryBeforeLoad,
        memoryRSSAfterLoadBytes: memoryAfterLoad,
        memoryRSSAfterGenerateBytes: memoryAfterGenerate,
        memoryRSSDeltaLoadBytes: deltaLoadBytes
    )

    return BenchmarkRow(
        status: "passed",
        runtime: "coreml",
        variant: variant,
        coremlDir: coremlDir.path,
        mlxModelDir: nil,
        modelPath: modelPath.path,
        inference: inference,
        error: nil,
        effectiveContextLength: effectiveContext
    )
}

func runAsync<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<T, Error>?
    Task {
        do {
            result = .success(try await operation())
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result else {
        throw BenchError.message("async operation produced no result")
    }
    return try result.get()
}

func runMLXBenchmark(config: CLIConfig) throws -> BenchmarkRow {
    let modelDir = config.mlxModelDir.standardizedFileURL

    return try runAsync {
        await HunyuanSupport.ensureRegistered()

        let translationStart = Date()
        let memoryBeforeLoad = currentRSSBytes()

        let loadStart = Date()
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(directory: modelDir)
        )
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let memoryAfterLoad = currentRSSBytes()

        let prompt = buildPrompt(targetLanguage: config.targetLanguage, sourceText: config.sourceText)
        let input = UserInput(
            chat: [
                .system(config.systemPrompt),
                .user(prompt),
            ]
        )

        let stream = try await container.perform { (context: ModelContext) in
            let lmInput = try await context.processor.prepare(input: input)
            let parameters = GenerateParameters(
                maxTokens: config.maxNewTokens,
                kvBits: nil,
                kvGroupSize: 64,
                quantizedKVStart: 0,
                temperature: 0.7,
                topP: 0.6,
                repetitionPenalty: 1.05,
                repetitionContextSize: 20
            )

            return try MLXLMCommon.generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )
        }

        var generatedText = ""
        var completionInfo: GenerateCompletionInfo?

        for await item in stream {
            switch item {
            case .chunk(let chunk):
                generatedText.append(chunk)
            case .info(let info):
                completionInfo = info
            case .toolCall:
                break
            }
        }

        if generatedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw BenchError.message("mlx generated empty text")
        }

        let memoryAfterGenerate = currentRSSBytes()
        let translationTotalSeconds = Date().timeIntervalSince(translationStart)

        let deltaLoadBytes: Int64?
        if let before = memoryBeforeLoad, let after = memoryAfterLoad {
            deltaLoadBytes = after - before
        } else {
            deltaLoadBytes = nil
        }

        let inference = InferenceMetrics(
            variant: modelDir.lastPathComponent,
            computeUnit: nil,
            compiledMaterialized: nil,
            statefulRuntime: nil,
            stateError: nil,
            loadSeconds: round(loadSeconds * 1000.0) / 1000.0,
            prefillSeconds: nil,
            firstTokenLatencySeconds: completionInfo.map { round($0.promptTime * 1000.0) / 1000.0 },
            generateSeconds: completionInfo.map { round($0.generateTime * 1000.0) / 1000.0 },
            translationTotalSeconds: round(translationTotalSeconds * 1000.0) / 1000.0,
            endToEndSeconds: round(translationTotalSeconds * 1000.0) / 1000.0,
            promptTokens: completionInfo?.promptTokenCount,
            outputTokens: completionInfo?.generationTokenCount,
            memoryRSSBeforeLoadMB: bytesToMB(memoryBeforeLoad),
            memoryRSSAfterLoadMB: bytesToMB(memoryAfterLoad),
            memoryRSSAfterGenerateMB: bytesToMB(memoryAfterGenerate),
            memoryRSSDeltaLoadMB: bytesToMB(deltaLoadBytes),
            memoryRSSBeforeLoadBytes: memoryBeforeLoad,
            memoryRSSAfterLoadBytes: memoryAfterLoad,
            memoryRSSAfterGenerateBytes: memoryAfterGenerate,
            memoryRSSDeltaLoadBytes: deltaLoadBytes
        )

        return BenchmarkRow(
            status: "passed",
            runtime: "mlx",
            variant: modelDir.lastPathComponent,
            coremlDir: nil,
            mlxModelDir: modelDir.path,
            modelPath: modelDir.path,
            inference: inference,
            error: nil,
            effectiveContextLength: nil
        )
    }
}

func parseVariants(raw: String) -> [String] {
    raw.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
}

func discoverCoreMLVariants(coremlRoot: URL, artifactStem: String) -> [String] {
    let prefix = artifactStem + "-"
    guard let children = try? fileManager.contentsOfDirectory(at: coremlRoot, includingPropertiesForKeys: nil) else {
        return []
    }

    var variants: [String] = []
    for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: child.path, isDirectory: &isDir), isDir.boolValue else {
            continue
        }
        let manifest = child.appendingPathComponent("translation-manifest.json")
        guard fileManager.fileExists(atPath: manifest.path) else {
            continue
        }

        let name = child.lastPathComponent
        if name.hasPrefix(prefix) {
            variants.append(String(name.dropFirst(prefix.count)))
        }
    }
    return variants
}

func resolveCoreMLDir(variant: String, coremlRoot: URL, artifactStem: String) throws -> URL {
    let dir = coremlRoot.appendingPathComponent("\(artifactStem)-\(variant)", isDirectory: true)
    var isDir: ObjCBool = false
    if fileManager.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue {
        return dir
    }
    throw BenchError.message("missing coreml dir for variant=\(variant): \(dir.path)")
}

func rowLabel(_ row: BenchmarkRow) -> String {
    return "\(row.runtime):\(row.variant)"
}

func summaryRow(from row: BenchmarkRow) -> SummaryModelRow {
    let inf = row.inference
    return SummaryModelRow(
        runtime: row.runtime,
        variant: row.variant,
        label: rowLabel(row),
        coremlDir: row.coremlDir,
        mlxModelDir: row.mlxModelDir,
        modelPath: row.modelPath,
        statefulRuntime: inf?.statefulRuntime,
        loadSeconds: inf?.loadSeconds,
        translationTotalSeconds: inf?.translationTotalSeconds,
        generateSeconds: inf?.generateSeconds,
        memoryRSSBeforeLoadMB: inf?.memoryRSSBeforeLoadMB,
        memoryRSSAfterLoadMB: inf?.memoryRSSAfterLoadMB,
        memoryRSSDeltaLoadMB: inf?.memoryRSSDeltaLoadMB,
        memoryRSSAfterGenerateMB: inf?.memoryRSSAfterGenerateMB,
        promptTokens: inf?.promptTokens,
        outputTokens: inf?.outputTokens
    )
}

func pickBest(rows: [BenchmarkRow], metric: KeyPath<InferenceMetrics, Double?>) -> BestMetric? {
    let candidates: [(Double, BenchmarkRow)] = rows.compactMap { row in
        guard row.status == "passed", let value = row.inference?[keyPath: metric] else {
            return nil
        }
        return (value, row)
    }

    guard let best = candidates.min(by: { $0.0 < $1.0 }) else {
        return nil
    }

    return BestMetric(
        runtime: best.1.runtime,
        variant: best.1.variant,
        label: rowLabel(best.1),
        value: round(best.0 * 1_000_000.0) / 1_000_000.0
    )
}

func buildRanking(rows: [BenchmarkRow], metricName: String, metric: KeyPath<InferenceMetrics, Double?>) -> [RankedMetricEntry] {
    let candidates: [(Double, BenchmarkRow)] = rows.compactMap { row in
        guard row.status == "passed", let value = row.inference?[keyPath: metric] else {
            return nil
        }
        return (value, row)
    }

    let sorted = candidates.sorted(by: { $0.0 < $1.0 })
    var rankings: [RankedMetricEntry] = []
    rankings.reserveCapacity(sorted.count)

    for (index, item) in sorted.enumerated() {
        rankings.append(
            RankedMetricEntry(
                rank: index + 1,
                runtime: item.1.runtime,
                variant: item.1.variant,
                label: rowLabel(item.1),
                metric: metricName,
                value: round(item.0 * 1_000_000.0) / 1_000_000.0
            )
        )
    }

    return rankings
}

func writeJSON<T: Encodable>(_ value: T, to path: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(value)
    try data.write(to: path)
}

func writeMarkdownSummary(path: URL, summary: SummaryReport) throws {
    var lines: [String] = []
    lines.append("# Swift Runtime Model Benchmark Summary")
    lines.append("")
    lines.append("- generated_at: `\(summary.generatedAt)`")
    lines.append("- coreml_compute_unit: `\(summary.coremlComputeUnit)`")
    lines.append("- include_mlx: `\(summary.includeMLX)`")
    lines.append("- total_models: `\(summary.totalModels)`")
    lines.append("- success_count: `\(summary.successCount)`")
    lines.append("- failure_count: `\(summary.failureCount)`")
    lines.append("")
    lines.append("## Best")
    lines.append("")
    lines.append("- fastest_load: `\(String(describing: summary.fastestLoad))`")
    lines.append("- fastest_translation: `\(String(describing: summary.fastestTranslation))`")
    lines.append("- lowest_load_memory: `\(String(describing: summary.lowestLoadMemory))`")
    lines.append("")
    lines.append("## Per Model")
    lines.append("")
    lines.append("| runtime | variant | load_seconds | translation_total_seconds | memory_before_load_mb | memory_after_load_mb | memory_delta_load_mb |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for row in summary.modelRows {
        lines.append(
            "| \(row.runtime) | \(row.variant) | \(optionalDouble(row.loadSeconds)) | \(optionalDouble(row.translationTotalSeconds)) | \(optionalDouble(row.memoryRSSBeforeLoadMB)) | \(optionalDouble(row.memoryRSSAfterLoadMB)) | \(optionalDouble(row.memoryRSSDeltaLoadMB)) |"
        )
    }

    try lines.joined(separator: "\n").appending("\n").write(to: path, atomically: true, encoding: .utf8)
}

func optionalDouble(_ value: Double?) -> String {
    guard let value else {
        return ""
    }
    return String(value)
}

func run() throws {
    let config = try parseCLI()

    let coremlRoot = config.coremlRootDir.standardizedFileURL
    var variants: [String]
    if config.variantsRaw == "auto" {
        variants = discoverCoreMLVariants(coremlRoot: coremlRoot, artifactStem: config.artifactStem)
    } else {
        variants = parseVariants(raw: config.variantsRaw)
    }

    if variants.isEmpty, !config.includeMLX {
        throw BenchError.message("no benchmark targets found")
    }

    var rows: [BenchmarkRow] = []
    var tokenCache: [String: TokenizeResponse] = [:]

    for variant in variants {
        do {
            let coremlDir = try resolveCoreMLDir(variant: variant, coremlRoot: coremlRoot, artifactStem: config.artifactStem)
            let row = try runCoreMLBenchmark(
                variant: variant,
                coremlDir: coremlDir,
                config: config,
                tokenCache: &tokenCache
            )
            rows.append(row)
            print("[coreml] completed variant=\(variant)")
        } catch {
            let coremlDir = try? resolveCoreMLDir(variant: variant, coremlRoot: coremlRoot, artifactStem: config.artifactStem)
            let failure = BenchmarkRow(
                status: "failed",
                runtime: "coreml",
                variant: variant,
                coremlDir: coremlDir?.path,
                mlxModelDir: nil,
                modelPath: nil,
                inference: nil,
                error: String(describing: error),
                effectiveContextLength: nil
            )
            rows.append(failure)
            print("[coreml] failed variant=\(variant) error=\(error)")
            if !config.continueOnError {
                throw error
            }
        }
    }

    if config.includeMLX {
        do {
            var isDir: ObjCBool = false
            if !fileManager.fileExists(atPath: config.mlxModelDir.path, isDirectory: &isDir) || !isDir.boolValue {
                throw BenchError.message("mlx model dir does not exist: \(config.mlxModelDir.path)")
            }

            let row = try runMLXBenchmark(config: config)
            rows.append(row)
            print("[mlx] completed model=\(config.mlxModelDir.lastPathComponent)")
        } catch {
            let failure = BenchmarkRow(
                status: "failed",
                runtime: "mlx",
                variant: config.mlxModelDir.lastPathComponent,
                coremlDir: nil,
                mlxModelDir: config.mlxModelDir.path,
                modelPath: nil,
                inference: nil,
                error: String(describing: error),
                effectiveContextLength: nil
            )
            rows.append(failure)
            print("[mlx] failed error=\(error)")
            if !config.continueOnError {
                throw error
            }
        }
    }

    let generatedAt = iso8601Now()
    let successRows = rows.filter { $0.status == "passed" }
    let failureRows = rows.filter { $0.status != "passed" }

    let summary = SummaryReport(
        status: failureRows.isEmpty ? "completed" : "completed_with_failures",
        generatedAt: generatedAt,
        coremlComputeUnit: config.computeUnitName,
        includeMLX: config.includeMLX,
        coremlRootDir: coremlRoot.path,
        mlxModelDir: config.mlxModelDir.path,
        artifactStem: config.artifactStem,
        coremlVariants: variants,
        totalModels: rows.count,
        successCount: successRows.count,
        failureCount: failureRows.count,
        fastestLoad: pickBest(rows: successRows, metric: \.loadSeconds),
        fastestTranslation: pickBest(rows: successRows, metric: \.translationTotalSeconds),
        lowestLoadMemory: pickBest(rows: successRows, metric: \.memoryRSSAfterLoadMB),
        modelRows: successRows.map(summaryRow(from:)),
        failures: failureRows,
        rankings: [
            "by_translation_total_seconds": buildRanking(rows: successRows, metricName: "translationTotalSeconds", metric: \.translationTotalSeconds),
            "by_load_seconds": buildRanking(rows: successRows, metricName: "loadSeconds", metric: \.loadSeconds),
            "by_memory_rss_after_load_mb": buildRanking(rows: successRows, metricName: "memoryRSSAfterLoadMB", metric: \.memoryRSSAfterLoadMB),
        ]
    )

    let fullResults = FullResults(
        status: summary.status,
        generatedAt: generatedAt,
        config: FullResults.ConfigSnapshot(
            coremlComputeUnit: config.computeUnitName,
            sourceText: config.sourceText,
            targetLanguage: config.targetLanguage,
            systemPrompt: config.systemPrompt,
            maxNewTokens: config.maxNewTokens,
            requestedContextLength: config.requestedContextLength,
            coremlRootDir: coremlRoot.path,
            artifactStem: config.artifactStem,
            coremlVariants: variants,
            includeMLX: config.includeMLX,
            mlxModelDir: config.mlxModelDir.path,
            pythonExecutable: config.pythonExecutable
        ),
        rows: rows,
        summary: summary
    )

    try fileManager.createDirectory(at: config.resultsDir, withIntermediateDirectories: true)
    let resultsJSONPath = config.resultsDir.appendingPathComponent(config.resultsJSONName)
    let summaryJSONPath = config.resultsDir.appendingPathComponent(config.summaryJSONName)
    let summaryMDPath = config.resultsDir.appendingPathComponent(config.summaryMDName)

    try writeJSON(fullResults, to: resultsJSONPath)
    try writeJSON(summary, to: summaryJSONPath)
    try writeMarkdownSummary(path: summaryMDPath, summary: summary)

    let output: [String: Any] = [
        "status": summary.status,
        "results_json": resultsJSONPath.path,
        "summary_json": summaryJSONPath.path,
        "summary_md": summaryMDPath.path,
        "total_models": summary.totalModels,
        "success_count": summary.successCount,
        "failure_count": summary.failureCount,
        "fastest_load": summary.fastestLoad.map { ["runtime": $0.runtime, "variant": $0.variant, "label": $0.label, "value": $0.value] as [String: Any] } ?? NSNull(),
        "fastest_translation": summary.fastestTranslation.map { ["runtime": $0.runtime, "variant": $0.variant, "label": $0.label, "value": $0.value] as [String: Any] } ?? NSNull(),
        "lowest_load_memory": summary.lowestLoadMemory.map { ["runtime": $0.runtime, "variant": $0.variant, "label": $0.label, "value": $0.value] as [String: Any] } ?? NSNull(),
    ]

    let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
    if let text = String(data: data, encoding: .utf8) {
        print(text)
    }
}

do {
    try run()
} catch {
    fputs("ERROR: \(error)\n", stderr)
    exit(1)
}
