# Phase 0：构建基线记录

> 状态：阻塞于本机 Rust 工具链<br>
> 记录日期：2026-07-31<br>
> 关联方案：[中文 Embedding Runtime 改造实施方案](./中文Embedding-Runtime改造实施方案.md)

## 已验证事实

| 项目 | 结果 |
|---|---|
| 源码提交 | `e63bcb770f6f4ca90017ca394b6e95ed68ef1748` (`chore: add project git workflow`) |
| 目标模块 | `:sentence_embeddings` |
| AGP / Kotlin | `8.9.0` / `2.1.0` |
| NDK 声明 | `28.1.13356709` (r28b) |
| ONNX Runtime Android 声明 | `1.23.0` |
| Gradle wrapper | `8.11.1` |
| `:sentence_embeddings:assembleRelease` | 成功，但未依赖 Rust `cargoBuild` 任务 |

## AAR 制品检查

`assembleRelease` 生成了：

```text
sentence_embeddings/build/outputs/aar/sentence_embeddings-release.aar
```

| 属性 | 值 |
|---|---|
| 大小 | 15,482 bytes |
| SHA-256 | `8E55C0ADC492FDC89152B3E3F634C06AE4C524C4603907660A19AE49A886023C` |
| AAR 条目 | `AndroidManifest.xml`、`classes.jar` |
| Java/Kotlin 类 | `HFTokenizer`、`HFTokenizer.Result`、`SentenceEmbedding` |
| JNI 库 | **未包含** |

这不是可交付 AAR：其中缺少运行 `HFTokenizer` 所需的 `libhftokenizer.so`。因此，单独执行 `assembleRelease` 不能作为本项目的完整基线验证。

## Native 构建诊断

项目定义了 `cargoBuild`、`cargoBuildArm`、`cargoBuildArm64`、`cargoBuildX86` 和 `cargoBuildX86_64` 任务。执行：

```powershell
.\gradlew.bat :sentence_embeddings:cargoBuild --stacktrace
```

在 `:sentence_embeddings:cargoBuildArm` 失败，根因是：

```text
Cannot run program "rustc": CreateProcess error=2, 系统找不到指定的文件。
```

`rustup`、`cargo` 与 `rustc` 当前均不在系统 PATH。因此没有生成 `libhftokenizer.so`，AAR 也无法打入各 ABI 的 JNI 库。

## 后续前置条件

继续 Phase 0 前，需要安装并在新终端中确认 Rust stable 工具链（`rustup`、`cargo`、`rustc`），然后安装项目要求的 Android Rust targets：

```powershell
rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android
```

工具链就绪后依序执行：

```powershell
.\gradlew.bat :sentence_embeddings:cargoBuild --stacktrace
.\gradlew.bat :sentence_embeddings:assembleRelease --stacktrace
```

然后重新检查 AAR 是否包含：

```text
jni/arm64-v8a/libhftokenizer.so
jni/armeabi-v7a/libhftokenizer.so
jni/x86/libhftokenizer.so
jni/x86_64/libhftokenizer.so
```

> [!WARNING]
> 不要通过提交预编译 `.so`、跳过 `cargoBuild`，或改写 JNI 包名来绕过该前置条件。先得到可复现的 native 构建与完整 AAR，才进入 API 和 pooling 改造。
