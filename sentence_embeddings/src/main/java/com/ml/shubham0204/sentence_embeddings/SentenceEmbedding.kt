package com.ml.shubham0204.sentence_embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.LongBuffer
import java.util.EnumSet
import kotlin.math.max
import kotlin.math.sqrt

class SentenceEmbedding {
    private lateinit var hfTokenizer: HFTokenizer
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var useTokenTypeIds: Boolean = false
    private var outputTensorName: String = ""
    private var normalizeEmbedding: Boolean = false
    private var poolingStrategy = PoolingStrategy.MEAN
    private var dimensions: Int? = null
    private var queryPrefix: String? = null

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
    ) = withContext(Dispatchers.IO) {
        hfTokenizer = HFTokenizer(tokenizerBytes)
        ortEnvironment = OrtEnvironment.getEnvironment()
        val options =
            OrtSession.SessionOptions().apply {
                if (useFP16) {
                    addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED))
                }
                if (useXNNPack) {
                    addXnnpack(
                        mapOf(
                            "intra_op_num_threads" to "2",
                        ),
                    )
                }
            }
        ortSession = ortEnvironment.createSession(modelFilepath, options)
        this@SentenceEmbedding.useTokenTypeIds = useTokenTypeIds
        this@SentenceEmbedding.outputTensorName = outputTensorName
        this@SentenceEmbedding.normalizeEmbedding = normalizeEmbeddings
        this@SentenceEmbedding.poolingStrategy = poolingStrategy
        this@SentenceEmbedding.dimensions = dimensions
        this@SentenceEmbedding.queryPrefix = queryPrefix
        Log.d(SentenceEmbedding::class.simpleName, "Input Names: " + ortSession.inputNames.toList())
        Log.d(
            SentenceEmbedding::class.simpleName,
            "Output Names: " + ortSession.outputNames.toList(),
        )
    }

    suspend fun encode(
        sentence: String,
        purpose: EmbeddingPurpose = EmbeddingPurpose.PASSAGE,
    ): FloatArray =
        withContext(Dispatchers.IO) {
            val text = if (purpose == EmbeddingPurpose.QUERY && !queryPrefix.isNullOrBlank()) "$queryPrefix$sentence" else sentence
            val result = hfTokenizer.tokenize(text)
            val inputTensorMap = mutableMapOf<String, OnnxTensor>()
            val idsTensor =
                OnnxTensor.createTensor(
                    ortEnvironment,
                    LongBuffer.wrap(result.ids),
                    longArrayOf(1, result.ids.size.toLong()),
                )
            inputTensorMap["input_ids"] = idsTensor
            val attentionMaskTensor =
                OnnxTensor.createTensor(
                    ortEnvironment,
                    LongBuffer.wrap(result.attentionMask),
                    longArrayOf(1, result.attentionMask.size.toLong()),
                )
            inputTensorMap["attention_mask"] = attentionMaskTensor
            if (useTokenTypeIds) {
                val tokenTypeIdsTensor =
                    OnnxTensor.createTensor(
                        ortEnvironment,
                        LongBuffer.wrap(result.tokenTypeIds),
                        longArrayOf(1, result.tokenTypeIds.size.toLong()),
                    )
                inputTensorMap["token_type_ids"] = tokenTypeIdsTensor
            }
            val outputs = ortSession.run(inputTensorMap)
            val tokenEmbeddings3D = outputs.get(outputTensorName).orElseThrow {
                IllegalArgumentException("Missing output tensor: $outputTensorName")
            }.value as Array<Array<FloatArray>>
            val tokenEmbeddings = tokenEmbeddings3D[0]
            val pooledEmbedding = when (poolingStrategy) {
                PoolingStrategy.MEAN -> meanPooling(tokenEmbeddings, result.attentionMask)
                PoolingStrategy.CLS -> tokenEmbeddings.first().copyOf()
            }
            require(dimensions == null || pooledEmbedding.size == dimensions) {
                "Embedding dimension ${pooledEmbedding.size} does not match configured $dimensions"
            }
            return@withContext if (normalizeEmbedding) {
                normalize(pooledEmbedding)
            } else {
                pooledEmbedding
            }
        }

    private fun meanPooling(
        tokenEmbeddings: Array<FloatArray>,
        attentionMask: LongArray,
    ): FloatArray {
        var pooledEmbeddings = FloatArray(tokenEmbeddings[0].size) { 0f }
        var validTokenCount = 0

        tokenEmbeddings
            .filterIndexed { index, _ -> attentionMask[index] == 1L }
            .forEachIndexed { index, token ->
                validTokenCount++
                token.forEachIndexed { j, value ->
                    pooledEmbeddings[j] += value
                }
            }

        // Avoid division by zero
        val divisor = max(validTokenCount, 1)
        pooledEmbeddings = pooledEmbeddings.map { it / divisor }.toFloatArray()

        return pooledEmbeddings
    }

    // Function to normalize embeddings
    private fun normalize(embeddings: FloatArray): FloatArray {
        // Calculate the L2 norm (Euclidean norm)
        val norm = sqrt(embeddings.sumOf { it * it.toDouble() }).toFloat()
        // Normalize each embedding by dividing by the norm
        return embeddings.map { it / norm }.toFloatArray()
    }

    fun close() {
        ortSession.close()
        ortEnvironment.close()
        hfTokenizer.close()
    }
}
