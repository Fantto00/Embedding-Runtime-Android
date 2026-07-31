# OmniEdge 适配 Embedding Runtime AAR 方案

> 状态：POC 接入方案<br>
> 目标：在独立 OmniEdge 测试仓库验证中文 BGE 向量检索，不替换正式 384 维链路。

## 1. 责任边界

| 组件 | 责任 |
|---|---|
| Embedding Runtime AAR | tokenizer、ONNX 推理、CLS/MEAN pooling、归一化、输入输出校验与资源释放。 |
| OmniEdge | 模型下载与 SHA-256、模型版本、512 维 ObjectBox 索引、历史 Chunk 重编码、检索切换与回滚。 |

> [!IMPORTANT]
> AAR 不下载模型、不管理 ObjectBox，也不允许把 ONNX、tokenizer 或 Hugging Face Token 提交到 Git。

## 2. Phase 1：隔离测试仓库

1. 将 OmniEdge clone 到独立目录，例如 `D:\Android\OmniEdge-Bge-Poc`。
2. 使用新的应用数据或 ObjectBox 数据库；不得复用正式 384 维索引。
3. 记录测试仓库 HEAD、当前 AAR、ONNX Runtime 版本和 `Chunk.chunkEmbedding` 的维度。

## 3. Phase 2：接入 AAR

1. 在本仓库生成制品：

   ```powershell
   .\gradlew.bat :sentence_embeddings:assembleRelease
   ```

2. 将 `sentence_embeddings-release.aar` 放入 OmniEdge 的 `app/libs/`，替换旧 AAR；不要并存。
3. OmniEdge 显式声明 `com.microsoft.onnxruntime:onnxruntime-android:1.23.0`。
4. 保持 `HFTokenizer` 的既有 JNI 包名，不对 native 方法或 ABI 做迁移。

## 4. Phase 3：模型文件契约

OmniEdge 管理模型文件：

```text
filesDir/embedding-models/
└── bge-small-zh-v1.5-fp32/
    ├── model.onnx
    ├── tokenizer.json
    └── manifest.json
```

`manifest.json` 必须记录模型 ID/revision、ONNX 和 tokenizer SHA-256、512 维、`CLS`、L2 归一化、输出张量名和 query 前缀。模型校验失败时禁止切换活动索引。

## 5. Phase 4：适配层

仅由 OmniEdge 的 `SentenceEmbeddingProvider` 调用 AAR；ViewModel、ObjectBox 和业务 UseCase 不直接依赖 AAR：

```kotlin
val config = EmbeddingModelConfig(
    modelId = "BAAI/bge-small-zh-v1.5",
    dimensions = 512,
    pooling = PoolingStrategy.CLS,
    normalize = true,
    outputTensorName = "last_hidden_state",
    queryPrefix = "为这个句子生成表示以用于检索相关文章：",
)

embedder.initialize(modelFile.absolutePath, tokenizerFile.readBytes(), config)

val chunk = embedder.encode(chunkText, EmbeddingPurpose.PASSAGE)
val query = embedder.encode(queryText, EmbeddingPurpose.QUERY)
```

Passage 不加前缀；只有 Query 加检索前缀。

## 6. Phase 5：512 维索引迁移

禁止把 512 维 BGE 向量写入现有 384 维 HNSW 索引。测试仓库使用并行字段或新实体：

```kotlin
@HnswIndex(dimensions = 512)
var bgeEmbedding: FloatArray
```

步骤：

1. 保留旧 `chunkEmbedding` 及索引。
2. 写入 `bgeEmbedding` 和 `embeddingModelId`。
3. 全量重编码历史 Chunk。
4. 用 feature flag 选择旧索引或 BGE 索引。
5. 失败时切回旧索引，不删除旧向量。

## 7. 切换门槛与回滚

允许测试仓库默认使用 BGE 前，必须确认：512 维有限向量、正确的 Query/Passage 语义、完整重编码、无新旧混检，以及中文 Top-K 结果符合预期。

回滚只切换活动索引和模型版本；不在验证期间删除旧索引、旧模型或旧向量。

## 8. 下一步

clone 完 OmniEdge 后，先读取 `SentenceEmbeddingProvider`、`Chunk` 实体与 ObjectBox 配置，再将本方案落实为精确的文件级改造和迁移提交。
