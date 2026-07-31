package com.ml.shubham0204.sentence_embeddings

enum class PoolingStrategy { MEAN, CLS }

enum class EmbeddingPurpose { QUERY, PASSAGE }

data class EmbeddingModelConfig(
    val modelId: String,
    val dimensions: Int,
    val pooling: PoolingStrategy,
    val normalize: Boolean,
    val outputTensorName: String,
    val queryPrefix: String? = null,
)
