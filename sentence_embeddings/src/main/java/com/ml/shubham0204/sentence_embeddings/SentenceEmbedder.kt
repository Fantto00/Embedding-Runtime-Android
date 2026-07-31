package com.ml.shubham0204.sentence_embeddings

interface SentenceEmbedder : AutoCloseable {
    suspend fun initialize(modelPath: String, tokenizerBytes: ByteArray, config: EmbeddingModelConfig)

    suspend fun encode(text: String, purpose: EmbeddingPurpose = EmbeddingPurpose.PASSAGE): FloatArray
}
