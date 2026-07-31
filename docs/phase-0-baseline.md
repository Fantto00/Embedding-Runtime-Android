# Phase 0：构建基线记录

> 状态：已完成<br>
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
| Rust stable | `rustc 1.97.1` / `cargo 1.97.1` |
| 已安装 Android Rust targets | `aarch64-linux-android`、`armv7-linux-androideabi`、`i686-linux-android`、`x86_64-linux-android` |
| 完整构建 | `:sentence_embeddings:cargoBuild :sentence_embeddings:assembleRelease` 成功；标准 `assembleRelease` 也已验证会触发 native 构建 |

## AAR 制品检查

`assembleRelease` 生成了：

```text
sentence_embeddings/build/outputs/aar/sentence_embeddings-release.aar
```

| 属性 | 值 |
|---|---|
| 大小 | 5,468,731 bytes |
| SHA-256 | `19D138FB3F8304C67A25C199CC3ABE08831045DBEB4FF6BDB8E11A7CC006ED5D` |
| AAR 条目 | `AndroidManifest.xml`、`classes.jar`、四个 `jni/*/libhftokenizer.so` |
| Java/Kotlin 类 | `HFTokenizer`、`HFTokenizer.Result`、`SentenceEmbedding` |
| JNI 库 | `arm64-v8a`、`armeabi-v7a`、`x86`、`x86_64` 均已包含 |

上游的单独 `assembleRelease` 曾不依赖 Rust `cargoBuild`，会产生缺少 JNI 库的 15,482-byte 无效 AAR。现已将 `cargoBuild` 接入 `preBuild`；清理模块生成物后执行标准 `:sentence_embeddings:assembleRelease` 已验证会重新编译四个 ABI 并打入 AAR。

## Native 构建诊断

项目定义了 `cargoBuild`、`cargoBuildArm`、`cargoBuildArm64`、`cargoBuildX86` 和 `cargoBuildX86_64` 任务。首次执行 native 构建时因未安装 Rust 工具链而失败：

```powershell
.\gradlew.bat :sentence_embeddings:cargoBuild --stacktrace
```

在 `:sentence_embeddings:cargoBuildArm` 失败，根因是：

```text
Cannot run program "rustc": CreateProcess error=2, 系统找不到指定的文件。
```

安装 Rust stable 和四个 Android target 后，重新运行完整构建成功。Rust 编译仅报告上游 `rs-hf-tokenizer/src/lib.rs` 中 `jbyteArray` 未使用，以及 linker stdout 警告；没有编译错误。

## 可复现构建前置条件

在新环境中，先安装并确认 Rust stable 工具链（`rustup`、`cargo`、`rustc`），然后安装项目要求的 Android Rust targets：

```powershell
rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android
```

工具链就绪后执行：

```powershell
.\gradlew.bat :sentence_embeddings:assembleRelease --stacktrace
```

再检查 AAR 是否包含：

```text
jni/arm64-v8a/libhftokenizer.so
jni/armeabi-v7a/libhftokenizer.so
jni/x86/libhftokenizer.so
jni/x86_64/libhftokenizer.so
```

> [!WARNING]
> 不要通过提交预编译 `.so`、移除 `preBuild -> cargoBuild` 依赖，或改写 JNI 包名来绕过该前置条件。已确认可复现的 native 构建与完整 AAR 后，才进入 API 和 pooling 改造。
