# 中文 Embedding Runtime 改造实施方案

> 状态：方案评审版<br>
> 日期：2026-07-31<br>
> 目标仓库：`Embedding-Runtime-Android`<br>
> 关联交接资料：[OmniEdge 中文 Embedding AAR 开发交接文档](./OmniEdge中文Embedding-AAR开发交接文档.md)

## 1. 结论

改造目标可行，但它不是把 Demo 里的英文文案翻译成中文，而是把当前主要面向英文模型的封装改造成**可配置、可验证的 Android Embedding Runtime AAR**，第一目标模型为 `BAAI/bge-small-zh-v1.5`。

当前工程已经具备三个关键基础：

- `HFTokenizer` 能从外部传入的 `tokenizer.json` 创建 tokenizer；
- `SentenceEmbedding` 能从文件路径创建 ONNX Runtime session；
- `app` 模块已有“复制模型到私有目录后初始化”的示例链路。

但不能仅在 `app/Config.kt` 新增 `BGE_SMALL_ZH_V1_5` 条目后就视为完成。当前实现固定使用 mean pooling、通过 `outputs.get(0)` 读取输出、由调用方手工指定 `token_type_ids`，并且没有定义可重复关闭、失败回滚和并发编码的契约。它能作为 POC 起点，却不足以作为交付给 OmniEdge 的通用运行时。

> [!IMPORTANT]
> 本方案只改造本仓库的 AAR 与最小验证 Demo。模型下载、模型文件治理、ObjectBox 索引从 384 维迁移到 512 维、历史向量重建和 OmniEdge 业务接入均属于 OmniEdge 仓库的职责，不在本仓库实现。

## 2. 已核对的基线与文档差异

| 项目 | 当前源码事实 | 对方案的影响 |
|---|---|---|
| 核心实现 | `sentence_embeddings/.../SentenceEmbedding.kt` 固定执行 attention-mask-aware mean pooling | 必须抽出 `MEAN` 与 `CLS` 两种 pooling。BGE 中文模型不能沿用 mean 结果冒充 CLS embedding。 |
| 输入构建 | 代码依据 `useTokenTypeIds` 布尔值决定是否传入该张量 | 初始化时应以 `OrtSession.inputNames` 为准，配置只能定义策略，不能覆盖模型实际输入。 |
| 输出读取 | 代码使用 `outputs.get(0)`，没有使用传入的 `outputTensorName` | 必须按名称取得输出，并检查输出存在、rank、batch、序列长度和隐藏维度。 |
| 资源与并发 | `close()` 无幂等保护；Demo 会并发调用同一实例的 `encode()` | 第一版必须定义所有权和并发策略；在 tokenizer 线程安全性未证实前，优先串行化编码临界区。 |
| 运行时版本 | `sentence_embeddings/build.gradle.kts` 当前依赖 ONNX Runtime Android `1.23.0`、NDK r28b | 交接文档中提到的 `1.17.0` 与 README 中的 NDK r27c 是历史信息；实现和发布记录必须以实际锁定版本为准。 |
| Demo 模型 | `app` 的 `preBuild` 下载三个英文模型 | 不应把 BGE 中文模型二进制提交到源码仓库；需要单独的外部模型 POC 输入。 |

## 3. 目标、边界与成功标准

### 3.1 第一版目标

交付一个不携带模型权重的 AAR，使宿主应用能够：

1. 传入 ONNX 模型绝对路径、tokenizer 字节和模型配置；
2. 使用 `MEAN` 或 `CLS` pooling 生成固定维度向量；
3. 用 `EmbeddingPurpose.QUERY` 与 `EmbeddingPurpose.PASSAGE` 区分 query 指令前缀；
4. 对 BGE 中文模型产出 512 维、L2 归一化后的向量；
5. 对 MiniLM 保持可回归的 384 维 mean-pooling 行为；
6. 在错误模型、错误 tokenizer、错误张量名、错误形状、重复关闭和并发调用时给出明确且可测试的结果。

### 3.2 非目标

- 不翻译模型本身、不训练或微调 BGE；
- 不将 ONNX、`tokenizer.json` 或 Hugging Face Token 提交到仓库；
- 不做运行时模型下载器、模型升级器或 SHA-256 管理器；
- 不改动 OmniEdge 的 ObjectBox、Chunk、RAG、OCR、ASR 或索引迁移；
- 不制作包含 ONNX Runtime 的 fat AAR；
- 不在未证明必要前升级 ONNX Runtime、重编 JNI tokenizer、删除 ABI 或加入 NNAPI/XNNPACK 性能承诺；
- 不把 Demo 界面中文翻译与 embedding 语义改造绑为同一个交付。若需要 UI 本地化，应另开一个资源化改动。

### 3.3 完成判定

只有下列条件同时满足，才可以宣布第一版完成：

- `bge-small-zh-v1.5` 在 arm64-v8a 真机上加载并输出 512 维有限浮点数；
- BGE 的 query 使用指定前缀、passage 不使用前缀，且 CLS + L2 归一化行为得到验证；
- MiniLM 的 384 维 mean-pooling 回归通过；
- JVM 单元测试、Android 仪器测试、release/R8 AAR 构建和真实模型 POC 均通过；
- 发布物包含 AAR、SHA-256、ABI 列表、依赖版本、支持模型配置、API 示例、许可证与已知限制；
- 没有模型二进制、构建产物、凭据或本地绝对路径进入 Git。

## 4. 推荐的目标契约

第一版在现有 `HFTokenizer` 之上增加新的稳定 API；不要在首轮改造中修改 JNI 绑定的包名或 native 方法签名。

```kotlin
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

interface SentenceEmbedder : AutoCloseable {
    suspend fun initialize(
        modelPath: String,
        tokenizerBytes: ByteArray,
        config: EmbeddingModelConfig,
    )

    suspend fun encode(text: String, purpose: EmbeddingPurpose): FloatArray

    override fun close()
}
```

实现层可保留旧 `SentenceEmbedding` 作为临时兼容外观，或在没有下游消费者时直接替换为新 API；做出选择前先记录现有调用方。无论哪种方式，JNI 相关的 `HFTokenizer` 暂时保持其现有包名和原生接口。

### BGE 中文配置基线

```kotlin
val BgeSmallZhV15 = EmbeddingModelConfig(
    modelId = "BAAI/bge-small-zh-v1.5",
    dimensions = 512,
    pooling = PoolingStrategy.CLS,
    normalize = true,
    outputTensorName = "last_hidden_state",
    queryPrefix = "为这个句子生成表示以用于检索相关文章：",
)
```

这份配置是实现目标，不是未经验证的模型事实。开始真实 POC 前，必须从实际使用的模型 revision 和 ONNX session 元数据确认输入名、输出名、维度、token type 要求与推荐 query 指令。

## 5. 分阶段实施计划

每个阶段都应作为一个可独立评审、可回滚、可验证的提交单元。后续真正修改代码时，提交前只暂存该阶段涉及的路径，不使用 `git add .` 或 `git add -A`。

### Phase 0：冻结基线与消除文档漂移

**目的：** 在改动前证明上游基础可构建，并把后续判断建立在当前源码而非历史描述上。

1. 记录当前仓库 HEAD、上游来源、`LICENSE`、Gradle/AGP/Kotlin/Rust/NDK 与 ONNX Runtime 的实际版本。
2. 运行不触发 Demo 模型下载的最小构建：`.\gradlew.bat :sentence_embeddings:assembleRelease --stacktrace`。
3. 解包生成的 AAR，记录 `classes.jar` 中公开类、`jni/` ABI 和 native 库名称。
4. 记录构建是否依赖本机 NDK/Rust 目标；若失败，保留完整错误和环境缺口，不通过“重写 tokenizer”绕过问题。
5. 更新或补充基线记录：明确 `sentence_embeddings` 的实际 ONNX Runtime 版本为当前构建文件所声明的版本，并标记 README/交接文档中的冲突信息。

**验收：** 有一份可复现的基线命令与结果；不存在未解释的工具链差异。

**暂停条件：** 原始 AAR 无法构建、许可证不适合 fork/分发，或构建要求修改 native tokenizer/NDK 版本。出现任一情况，先给出证据和选项，再继续。

### Phase 1：定义库边界和稳定 API

**目的：** 让 AAR 的职责仅限于 tokenizer、ONNX 推理、pooling、归一化和资源管理。

1. 建立公共配置类型：`EmbeddingModelConfig`、`PoolingStrategy`、`EmbeddingPurpose`、`SentenceEmbedder` 与受控的 `EmbeddingException` 层级。
2. 设计实现类（例如 `OnnxSentenceEmbedder`），将输入预处理、ONNX 张量创建、输出校验、pooling 和向量计算拆为内部单元，避免所有逻辑继续堆在 `SentenceEmbedding`。
3. 规定初始化状态机：`New -> Initializing -> Ready -> Closed`。初始化失败必须关闭已创建的 tokenizer/session，不能留下半初始化对象。
4. 规定 `close()` 幂等；实例拥有并关闭 session/tokenizer/临时张量，但不错误关闭共享的 `OrtEnvironment`。
5. 规定并发策略：第一版优先用实例级 `Mutex` 串行化 tokenizer + session 的访问；若后续经真实压测和 native 代码审计证明线程安全，再讨论放宽。
6. 明确 dispatcher 责任：公开 API 为 `suspend`，执行 I/O/推理时使用后台 dispatcher，库内不使用 `runBlocking`。

**验收：** 公共 API 无 OmniEdge 类型、无模型下载职责、无 ObjectBox 依赖；状态转换和异常类型可在不加载真实模型的测试中覆盖。

### Phase 2：参数化现有推理内核

**目的：** 消除当前只适配一种模型输出结构的硬编码。

1. 初始化后读取并缓存 `OrtSession.inputNames`、`outputNames` 与必要元数据。
2. 始终构造 `input_ids`、`attention_mask`；仅当 session 声明 `token_type_ids` 时构造该张量。发现模型需要未知输入时失败并抛出 `UnsupportedModelInput`。
3. 以 `outputTensorName` 按名称读取结果，不以位置 `get(0)` 读取；缺失时抛出 `MissingOutputTensor` 并列出实际输出名。
4. 校验输出为预期浮点三维张量、batch 为 1、序列长度可与 attention mask 对齐、hidden size 等于 `config.dimensions`；不匹配时抛出带实际形状的异常。
5. 实现并单测：
   - `MEAN`：只对 attention mask 为 1 的 token 做平均；
   - `CLS`：取 batch 的第 0 个 token 向量；
   - `L2 normalize`：拒绝或以明确策略处理零向量，绝不产生 NaN/Infinity。
6. 对每次 `encode()` 使用 `use`/`finally` 关闭输入 `OnnxTensor` 和推理结果，确认异常路径也会释放。
7. 保留当前 `useFP16`/`useXNNPack` 的行为但不承诺 BGE 加速；将加速选择从模型语义配置中分离，默认 CPU 基线。

**验收：** 不依赖真实 ONNX 的纯 Kotlin 测试覆盖 pooling、归一化、query 前缀、维度/形状校验；所有失败信息可直接定位到模型契约问题。

### Phase 3：建立中文模型 POC

**目的：** 用真实 `BAAI/bge-small-zh-v1.5` 证明 tokenizer、ONNX 导出、pooling 与 query 语义在 Android 上一致。

1. 从模型官方来源取得明确 revision 的 FP32 ONNX、`tokenizer.json`、LICENSE 与 SHA-256；模型文件放在仓库外的受控测试目录。
2. 在桌面参考环境生成少量无敏感信息的 query、正样本 passage、负样本 passage 的基准结果，保存模型 revision、导出方式和推理配置。
3. 先用 ONNX Runtime session 打印 Android 侧实际输入名、输出名、shape，核对 Phase 2 的模型配置；不一致时修正配置或停止，不猜测。
4. 在最小 Android 验证 App 中将模型复制到 `filesDir`，使用 `BgeSmallZhV15` 分别编码 query 与 passage。
5. 验证：维度为 512、所有元素有限、L2 范数接近 1、同一文本跨运行稳定、query-positive 的相似度高于 query-negative，且 Top-K 排序与参考结果一致或误差可解释。
6. 在 arm64-v8a 真机记录模型加载、首次编码、热编码耗时和峰值内存；该数据只用于建立基线，不在没有对照实验时宣称优化收益。

**验收：** POC 可复现，且不把模型二进制或下载凭据写入仓库。

**暂停条件：** 模型需要当前 ONNX Runtime 不支持的算子、ONNX session 元数据和模型约定冲突、内存/延迟不能满足宿主要求，或必须修改 JNI tokenizer。应报告完整错误、可选路径（升级 runtime、重新导出、量化、重建 native）及其兼容性和许可证影响，等待确认。

### Phase 4：兼容性、异常与生命周期测试

**目的：** 让“可运行”升级为“可交付”。

1. JVM 单元测试：手工可计算的小张量验证 mean/CLS、mask、L2、零向量、query/passsage 前缀、形状与维度校验。
2. Android 仪器测试：验证 native 库加载、tokenizer 初始化/关闭、ONNX session 初始化、真实模型编码、AAR ABI 打包和 release/R8。
3. 最少覆盖下列负向路径：模型不存在、tokenizer 损坏、缺少目标输出、非三维输出、维度不符、`token_type_ids` 有/无、空串、超长输入、初始化失败、重复 `close()`、关闭后编码、并发 `encode()`。
4. 为 MiniLM 保留回归夹具：`MEAN + 384`；为 BGE 保留中文 golden 数据：`CLS + 512 + normalize + query prefix`。允许浮点容差，不能要求跨 runtime 完全逐位相等。
5. 检查 Demo 在切换模型时先关闭旧 embedder 后创建新实例，避免复用已关闭对象或泄漏 session。

**验收：** 每条失败路径都有确定异常或预期行为；release 构建不依赖下载模型的副作用。

### Phase 5：Demo、文档与发布制品

**目的：** 使下游可以独立接入，而不是只能阅读源码猜测用法。

1. 将 Demo 定位为验证器：提供外部模型文件路径/测试注入入口，不把中文模型纳入 Git；若要展示中文 UI，单独将硬编码文字提取至 `values/strings.xml`，不混入核心 AAR 改动。
2. 编写 API 使用示例，展示初始化、`QUERY`、`PASSAGE`、`close()` 与错误处理；示例不得包含真实 token、个人路径或模型二进制。
3. 编写 `SUPPORTED_MODELS.md`：模型 ID、revision、维度、pooling、归一化、输出张量名、query 前缀、已验证 ABI 和限制。
4. 生成 release AAR、SHA-256、大小、ABI 清单、依赖版本、构建命令、测试结果、许可证/NOTICE 与性能基线。
5. 在干净宿主工程中验证只引入 AAR 和声明的 ONNX Runtime 依赖即可编译；不得同时打入旧、新 AAR，以避免 duplicate class 或重复 `libhftokenizer.so`。

**验收：** 一个不了解本仓库内部结构的 Android 调用方可依据交付物完成初始化、编码和资源释放。

## 6. 建议的提交顺序

| 顺序 | 建议提交 | 可独立回答的评审问题 |
|---:|---|---|
| 1 | `docs: record embedding runtime baseline` | 当前上游和工具链能否复现？ |
| 2 | `refactor: introduce configurable embedding runtime api` | 公共边界、状态和异常契约是否稳定？ |
| 3 | `feat: support configurable pooling and output validation` | AAR 是否正确支持 MEAN/CLS、模型输入输出和 L2？ |
| 4 | `test: cover embedding runtime contracts` | 纯 Kotlin 与 Android 行为是否被覆盖？ |
| 5 | `feat: verify Chinese BGE embedding runtime` | BGE 中文模型能否在真机正确运行？ |
| 6 | `docs: publish embedding runtime integration guide` | 下游是否能独立接入和验收？ |

若 API 提取和推理内核改造互相依赖，应合并为一个完整可构建的提交，不能为了拆分而留下不可用的中间状态。

## 7. 风险与决策点

| 风险 | 处理方式 |
|---|---|
| “中文化”被误解为 UI 翻译 | 先以本方案的 embedding 语义目标为准；UI 本地化单独立项。 |
| 模型导出张量名与文档不同 | 以实际 `OrtSession` 元数据为准，写入模型配置和测试记录。 |
| BGE 与现有 MiniLM 的 pooling 语义不同 | 把 pooling 固化为每个模型的显式配置，并保留两套回归。 |
| JNI 包名变更导致 `UnsatisfiedLinkError` | 第一版保留 `HFTokenizer` 现有 JNI 边界，仅在其外层建立新 API。 |
| ONNX Runtime 版本/模型算子不兼容 | 先真实 POC；升级 runtime 或重新导出必须有单独评审和回归。 |
| 384 与 512 维向量混检 | 明确为 OmniEdge 迁移责任；本 AAR 只报告配置维度并做输出校验。 |
| 模型文件、许可证或凭据泄漏 | 模型放仓库外，交付时记录来源、revision、LICENSE 和 SHA-256；禁止提交令牌。 |
| 性能结论被过度承诺 | 先收集 CPU 真机基线，再以数据决定是否评估 XNNPACK/NNAPI。 |

## 8. 下一步

推荐从 **Phase 0** 开始：先复现当前 `:sentence_embeddings` 的 release AAR、记录事实基线并确认工具链。基线通过后，再进入 API/推理内核改造；不要先下载或集成 BGE 模型，更不要修改 OmniEdge 的 512 维索引。
