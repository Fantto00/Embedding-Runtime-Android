package com.ml.shubham0204.sentence_embeddings

sealed class EmbeddingException(message: String, cause: Throwable? = null) : RuntimeException(message, cause) {
    class ModelLoadFailed(cause: Throwable) : EmbeddingException("Unable to load embedding model", cause)
    class UnsupportedModelInput(inputs: Set<String>) : EmbeddingException("Unsupported model inputs: $inputs")
    class MissingOutputTensor(name: String, outputs: Set<String>) : EmbeddingException("Missing output tensor '$name'. Available: $outputs")
    class UnexpectedOutputShape(detail: String) : EmbeddingException(detail)
    class DimensionMismatch(expected: Int, actual: Int) : EmbeddingException("Expected $expected dimensions, got $actual")
    class Closed : EmbeddingException("Embedding runtime is closed")
    class NotInitialized : EmbeddingException("Embedding runtime is not initialized")
}
