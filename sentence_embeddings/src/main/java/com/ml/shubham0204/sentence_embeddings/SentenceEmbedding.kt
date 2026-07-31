package com.ml.shubham0204.sentence_embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.nio.LongBuffer
import java.util.EnumSet
import kotlin.math.sqrt

class SentenceEmbedding : SentenceEmbedder {
    private val mutex = Mutex()
    private var tokenizer: HFTokenizer? = null
    private var session: OrtSession? = null
    private var config: EmbeddingModelConfig? = null
    private var closed = false

    override suspend fun initialize(modelPath: String, tokenizerBytes: ByteArray, config: EmbeddingModelConfig) {
        initializeInternal(modelPath, tokenizerBytes, config, false, false)
    }

    suspend fun init(
        modelFilepath: String,
        tokenizerBytes: ByteArray,
        useTokenTypeIds: Boolean,
        outputTensorName: String,
        useFP16: Boolean = false,
        useXNNPack: Boolean = false,
        normalizeEmbeddings: Boolean,
        poolingStrategy: PoolingStrategy = PoolingStrategy.MEAN,
        dimensions: Int? = null,
        queryPrefix: String? = null,
    ) {
        initializeInternal(
            modelFilepath,
            tokenizerBytes,
            EmbeddingModelConfig("legacy", dimensions ?: -1, poolingStrategy, normalizeEmbeddings, outputTensorName, queryPrefix),
            useFP16,
            useXNNPack,
        )
    }

    private suspend fun initializeInternal(modelPath: String, tokenizerBytes: ByteArray, modelConfig: EmbeddingModelConfig, useFP16: Boolean, useXNNPack: Boolean) = withContext(Dispatchers.IO) {
        mutex.withLock {
            check(!closed) { throw EmbeddingException.Closed() }
            closeResources()
            try {
                val environment = OrtEnvironment.getEnvironment()
                val options = OrtSession.SessionOptions().apply {
                    if (useFP16) addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED))
                    if (useXNNPack) addXnnpack(mapOf("intra_op_num_threads" to "2"))
                }
                val createdSession = try { environment.createSession(modelPath, options) } finally { options.close() }
                validateInputs(createdSession.inputNames)
                tokenizer = HFTokenizer(tokenizerBytes)
                session = createdSession
                config = modelConfig
            } catch (error: Throwable) {
                closeResources()
                throw EmbeddingException.ModelLoadFailed(error)
            }
        }
    }

    override suspend fun encode(text: String, purpose: EmbeddingPurpose): FloatArray = withContext(Dispatchers.IO) {
        mutex.withLock {
            check(!closed) { throw EmbeddingException.Closed() }
            val activeSession = session ?: throw EmbeddingException.NotInitialized()
            val activeTokenizer = tokenizer ?: throw EmbeddingException.NotInitialized()
            val activeConfig = config ?: throw EmbeddingException.NotInitialized()
            val input = if (purpose == EmbeddingPurpose.QUERY && !activeConfig.queryPrefix.isNullOrBlank()) "${activeConfig.queryPrefix}$text" else text
            val tokens = activeTokenizer.tokenize(input)
            val tensors = createInputs(activeSession, tokens)
            try {
                val outputs = activeSession.run(tensors)
                try {
                    val value = outputs.get(activeConfig.outputTensorName).orElseThrow {
                        EmbeddingException.MissingOutputTensor(activeConfig.outputTensorName, activeSession.outputNames)
                    }.value as? Array<Array<FloatArray>> ?: throw EmbeddingException.UnexpectedOutputShape("Expected rank-3 float output")
                    val rows = value.singleOrNull() ?: throw EmbeddingException.UnexpectedOutputShape("Expected batch size 1")
                    if (rows.size != tokens.attentionMask.size || rows.isEmpty()) throw EmbeddingException.UnexpectedOutputShape("Output sequence length does not match attention mask")
                    val embedding = when (activeConfig.pooling) {
                        PoolingStrategy.MEAN -> meanPooling(rows, tokens.attentionMask)
                        PoolingStrategy.CLS -> rows.first().copyOf()
                    }
                    if (activeConfig.dimensions > 0 && embedding.size != activeConfig.dimensions) throw EmbeddingException.DimensionMismatch(activeConfig.dimensions, embedding.size)
                    if (activeConfig.normalize) normalize(embedding) else embedding
                } finally { outputs.close() }
            } finally { tensors.values.forEach { it.close() } }
        }
    }

    override fun close() = synchronized(this) {
        if (closed) return
        closed = true
        closeResources()
    }

    private fun createInputs(session: OrtSession, tokens: HFTokenizer.Result): MutableMap<String, OnnxTensor> {
        val environment = OrtEnvironment.getEnvironment()
        val shape = longArrayOf(1, tokens.ids.size.toLong())
        return mutableMapOf<String, OnnxTensor>().apply {
            put("input_ids", OnnxTensor.createTensor(environment, LongBuffer.wrap(tokens.ids), shape))
            put("attention_mask", OnnxTensor.createTensor(environment, LongBuffer.wrap(tokens.attentionMask), shape))
            if ("token_type_ids" in session.inputNames) put("token_type_ids", OnnxTensor.createTensor(environment, LongBuffer.wrap(tokens.tokenTypeIds), shape))
        }
    }

    private fun validateInputs(inputs: Set<String>) {
        val required = setOf("input_ids", "attention_mask")
        if (!inputs.containsAll(required) || (inputs - required - "token_type_ids").isNotEmpty()) throw EmbeddingException.UnsupportedModelInput(inputs)
    }

    private fun meanPooling(rows: Array<FloatArray>, mask: LongArray): FloatArray {
        val result = FloatArray(rows.first().size)
        var count = 0
        rows.forEachIndexed { index, row -> if (mask[index] == 1L) { count++; row.forEachIndexed { dimension, value -> result[dimension] += value } } }
        if (count == 0) throw EmbeddingException.UnexpectedOutputShape("Attention mask has no valid tokens")
        return result.map { it / count }.toFloatArray()
    }

    private fun normalize(vector: FloatArray): FloatArray {
        val norm = sqrt(vector.sumOf { it * it.toDouble() }).toFloat()
        if (norm == 0f || !norm.isFinite()) throw EmbeddingException.UnexpectedOutputShape("Cannot normalize a zero or non-finite vector")
        return vector.map { it / norm }.toFloatArray()
    }

    private fun closeResources() {
        session?.close(); session = null
        tokenizer?.close(); tokenizer = null
        config = null
    }
}
