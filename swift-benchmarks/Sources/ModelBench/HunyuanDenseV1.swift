//
//  HunyuanDenseV1.swift
//  translator-new
//
//  Created by Codex on 2026/4/17.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

nonisolated private func makeHunyuanDenseV1Model(from configuration: Data) throws -> any LanguageModel {
    let config = try JSONDecoder().decode(HunyuanDenseV1Configuration.self, from: configuration)
    return HunyuanDenseV1Model(config)
}

enum HunyuanSupport {
    private actor RegistrationState {
        private var didRegister = false

        func ensureRegistered() async {
            guard !didRegister else {
                return
            }

            await LLMTypeRegistry.shared.registerModelType(
                HunyuanDenseV1Configuration.modelType,
                creator: makeHunyuanDenseV1Model(from:)
            )
            didRegister = true
        }
    }

    private static let registrationState = RegistrationState()

    nonisolated static func ensureRegistered() async {
        await registrationState.ensureRegistered()
    }
}

private nonisolated final class HunyuanDynamicNTKScalingRoPE: Module {
    let dimensions: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    var base: Float?
    let scale: Float
    let ropeType: String
    let ropeScaling: [String: StringOrNumber]?
    var freqs: MLXArray?

    @available(*, unavailable)
    nonisolated override init() {
        fatalError("Use init(dimensions:maxPositionEmbeddings:traditional:base:scale:ropeType:ropeScaling:)")
    }

    nonisolated init(
        dimensions: Int,
        maxPositionEmbeddings: Int?,
        traditional: Bool = false,
        base: Float = 10_000,
        scale: Float = 1.0,
        ropeType: String = "default",
        ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.dimensions = dimensions
        self.maxPositionEmbeddings = maxPositionEmbeddings ?? 2048
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.ropeType = ropeType
        self.ropeScaling = ropeScaling
        super.init()
        computeFreqs()
    }

    nonisolated private func computeFreqs() {
        if ropeType != "llama3" {
            freqs = nil
            return
        }

        guard
            let ropeScaling,
            case .float(let factor) = ropeScaling["factor"],
            case .float(let lowFreqFactor) = ropeScaling["low_freq_factor"] ?? .float(1.0),
            case .float(let highFreqFactor) = ropeScaling["high_freq_factor"] ?? .float(4.0),
            case .float(let originalMaxPositionEmbeddings) =
                ropeScaling["original_max_position_embeddings"] ?? .float(8192),
            let base
        else {
            freqs = nil
            return
        }

        let lowFreqWavelen = originalMaxPositionEmbeddings / lowFreqFactor
        let highFreqWavelen = originalMaxPositionEmbeddings / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dimensions, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dimensions))
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen),
            frequencies * factor,
            frequencies
        )

        let isMediumFrequency = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )
        let smoothFactors =
            (originalMaxPositionEmbeddings / wavelens - lowFreqFactor)
            / (highFreqFactor - lowFreqFactor)
        let smoothFrequencies =
            frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        freqs = MLX.where(isMediumFrequency, smoothFrequencies, frequencies)
        self.base = nil
    }

    nonisolated func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: base,
            scale: scale,
            offset: offset,
            freqs: freqs
        )
    }
}

private nonisolated final class HunyuanDenseV1Attention: Module {
    let config: HunyuanDenseV1Configuration
    let scale: Float
    let attentionHeads: Int
    let kvHeads: Int
    let headDimension: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "query_layernorm") var queryLayerNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyLayerNorm: RMSNorm?

    let rope: HunyuanDynamicNTKScalingRoPE

    @available(*, unavailable)
    nonisolated override init() {
        fatalError("Use init(_:)")
    }

    nonisolated init(_ config: HunyuanDenseV1Configuration) {
        self.config = config
        self.attentionHeads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDimension = config.resolvedHeadDimension
        self.scale = pow(Float(headDimension), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize,
            attentionHeads * headDimension,
            bias: config.attentionBias
        )
        _kProj.wrappedValue = Linear(
            config.hiddenSize,
            kvHeads * headDimension,
            bias: config.attentionBias
        )
        _vProj.wrappedValue = Linear(
            config.hiddenSize,
            kvHeads * headDimension,
            bias: config.attentionBias
        )
        _oProj.wrappedValue = Linear(
            attentionHeads * headDimension,
            config.hiddenSize,
            bias: config.attentionBias
        )

        if config.useQKNorm {
            _queryLayerNorm.wrappedValue = RMSNorm(
                dimensions: headDimension,
                eps: config.rmsNormEps
            )
            _keyLayerNorm.wrappedValue = RMSNorm(
                dimensions: headDimension,
                eps: config.rmsNormEps
            )
        } else {
            _queryLayerNorm.wrappedValue = nil
            _keyLayerNorm.wrappedValue = nil
        }

        let ropeType = {
            if case .string(let type) = config.ropeScaling?["type"] ?? config.ropeScaling?["rope_type"]
            {
                return type
            } else {
                return "default"
            }
        }()
        let ropeScale: Float = {
            if ropeType == "linear", let factor = config.ropeScaling?["factor"]?.asFloat() {
                return 1 / factor
            } else {
                return 1.0
            }
        }()

        self.rope = HunyuanDynamicNTKScalingRoPE(
            dimensions: headDimension,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            traditional: false,
            base: config.ropeTheta,
            scale: ropeScale,
            ropeType: ropeType,
            ropeScaling: config.ropeScaling
        )
    }

    nonisolated func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries
            .reshaped(batchSize, sequenceLength, attentionHeads, -1)
        keys = keys
            .reshaped(batchSize, sequenceLength, kvHeads, -1)
        values = values
            .reshaped(batchSize, sequenceLength, kvHeads, -1)

        if let queryLayerNorm {
            queries = queryLayerNorm(queries)
        }
        if let keyLayerNorm {
            keys = keyLayerNorm(keys)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return oProj(output)
    }
}

private nonisolated final class HunyuanDenseV1MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    @available(*, unavailable)
    nonisolated override init() {
        fatalError("Use init(_:)")
    }

    nonisolated init(_ config: HunyuanDenseV1Configuration) {
        _gateProj.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        _upProj.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias
        )
        _downProj.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
    }

    nonisolated func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private nonisolated final class HunyuanDenseV1DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: HunyuanDenseV1Attention
    @ModuleInfo(key: "mlp") var mlp: HunyuanDenseV1MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @available(*, unavailable)
    nonisolated override init() {
        fatalError("Use init(_:)")
    }

    nonisolated init(_ config: HunyuanDenseV1Configuration) {
        _selfAttention.wrappedValue = HunyuanDenseV1Attention(config)
        _mlp.wrappedValue = HunyuanDenseV1MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    nonisolated func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hiddenStates = x
        var residual = hiddenStates

        hiddenStates = inputLayerNorm(hiddenStates)
        hiddenStates = selfAttention(hiddenStates, mask: mask, cache: cache)
        hiddenStates = residual + hiddenStates

        residual = hiddenStates
        hiddenStates = postAttentionLayerNorm(hiddenStates)
        hiddenStates = mlp(hiddenStates)
        hiddenStates = residual + hiddenStates

        return hiddenStates
    }
}

private nonisolated final class HunyuanDenseV1ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [HunyuanDenseV1DecoderLayer]
    let norm: RMSNorm

    @available(*, unavailable)
    nonisolated override init() {
        fatalError("Use init(_:)")
    }

    nonisolated init(_ config: HunyuanDenseV1Configuration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        layers = (0..<config.hiddenLayers).map { _ in HunyuanDenseV1DecoderLayer(config) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    nonisolated func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = embedTokens(inputs)
        let mask = createAttentionMask(h: hiddenStates, cache: cache?.first)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[index])
        }

        return norm(hiddenStates)
    }
}

public nonisolated final class HunyuanDenseV1Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let model: HunyuanDenseV1ModelInner
    let configuration: HunyuanDenseV1Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    @available(*, unavailable)
    nonisolated public override init() {
        fatalError("Use init(_:)")
    }

    nonisolated public init(_ configuration: HunyuanDenseV1Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = (0..<configuration.hiddenLayers).map { _ in configuration.kvHeads }
        self.model = HunyuanDenseV1ModelInner(configuration)

        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    nonisolated public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let output = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(output)
        } else {
            return model.embedTokens.asLinear(output)
        }
    }

    nonisolated public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }

    nonisolated public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        do {
            _ = try tokenizer.applyChatTemplate(
                messages: [["role": "system", "content": "test"]]
            )
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }

    nonisolated public var loraLayers: [Module] {
        model.layers
    }
}

public struct HunyuanDenseV1Configuration: Codable, Sendable {
    nonisolated static let modelType = "hunyuan_v1_dense"

    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDimension: Int?
    var maxPositionEmbeddings: Int?
    var rmsNormEps: Float
    var ropeTheta: Float = 10_000
    var ropeScaling: [String: StringOrNumber]?
    var vocabularySize: Int
    var tieWordEmbeddings: Bool = true
    var attentionBias: Bool = false
    var mlpBias: Bool = false
    var useQKNorm: Bool = false
    var hiddenAct: String = "silu"

    nonisolated var resolvedHeadDimension: Int {
        headDimension ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case useQKNorm = "use_qk_norm"
        case hiddenAct = "hidden_act"
    }

    public nonisolated init(from decoder: Swift.Decoder) throws {
        let container: KeyedDecodingContainer<HunyuanDenseV1Configuration.CodingKeys> =
            try decoder.container(keyedBy: HunyuanDenseV1Configuration.CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? tieWordEmbeddings
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
            ?? attentionBias
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? mlpBias
        useQKNorm = try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? useQKNorm
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? hiddenAct

        if let ropeScaling {
            if ropeScaling["factor"] == nil {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling,
                    in: container,
                    debugDescription: "rope_scaling must contain 'factor'"
                )
            }

            if let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"] {
                if case .string = ropeType {
                    let options = [
                        StringOrNumber.string("linear"),
                        StringOrNumber.string("dynamic"),
                        StringOrNumber.string("llama3"),
                    ]
                    if !options.contains(ropeType) {
                        throw DecodingError.dataCorruptedError(
                            forKey: .ropeScaling,
                            in: container,
                            debugDescription:
                                "rope_scaling 'type' currently only supports 'linear', 'dynamic', or 'llama3'"
                        )
                    }
                }
            } else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling,
                    in: container,
                    debugDescription: "rope_scaling must contain either 'type' or 'rope_type'"
                )
            }
        }
    }
}
