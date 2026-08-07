package com.ml.shubham0204.sentence_embeddings

import org.json.JSONObject

class HFTokenizer(
    tokenizerBytes: ByteArray,
) : AutoCloseable {
    data class Result(
        val ids: LongArray = longArrayOf(),
        val attentionMask: LongArray = longArrayOf(),
        val tokenTypeIds: LongArray = longArrayOf(),
    )

    private var tokenizerPtr: Long = createTokenizer(tokenizerBytes)

    @Synchronized
    fun tokenize(text: String): Result {
        check(tokenizerPtr != 0L) { "Tokenizer is closed" }
        val output = tokenize(tokenizerPtr, text)
        val jsonObject = JSONObject(output)
        val idsArray = jsonObject.getJSONArray("ids")
        val ids = LongArray(idsArray.length())
        for (i in 0 until idsArray.length()) {
            ids[i] = (idsArray.get(i) as Int).toLong()
        }
        val attentionMaskArray = jsonObject.getJSONArray("attention_mask")
        val attentionMask = LongArray(attentionMaskArray.length())
        for (i in 0 until attentionMaskArray.length()) {
            attentionMask[i] = (attentionMaskArray.get(i) as Int).toLong()
        }
        val tokenTypeIdsArray = jsonObject.getJSONArray("token_type_ids")
        val tokenTypeIds = LongArray(tokenTypeIdsArray.length())
        for (i in 0 until tokenTypeIdsArray.length()) {
            tokenTypeIds[i] = (tokenTypeIdsArray.get(i) as Int).toLong()
        }
        return Result(ids, attentionMask, tokenTypeIds)
    }

    @Synchronized
    override fun close() {
        if (tokenizerPtr == 0L) return
        val pointer = tokenizerPtr
        tokenizerPtr = 0L
        deleteTokenizer(pointer)
    }

    private external fun createTokenizer(tokenizerBytes: ByteArray): Long

    private external fun tokenize(
        tokenizerPtr: Long,
        text: String,
    ): String

    private external fun deleteTokenizer(tokenizerPtr: Long)

    companion object {
        init {
            System.loadLibrary("hftokenizer")
        }
    }
}
