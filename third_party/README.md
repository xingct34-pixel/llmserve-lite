# 固定依赖与许可证

| 依赖 | 固定版本 | 许可 | 纳入本仓库 |
|---|---|---|---|
| cpp-httplib | v0.18.1 | MIT | 单头文件与原始LICENSE |
| nlohmann/json | v3.11.3 | MIT | 单头文件与原始LICENSE |
| llama.cpp | b6500 / a7a98e0fffed794396b3fbad4dcdbbc184963645 | MIT | 仅来源、校验信息和许可证副本，不提交完整源码 |
| Qwen2.5-0.5B-Instruct-GGUF | 9217f5db79a29953eb74d5343926648285ec7e67 | 官方仓库标注Apache-2.0 | 仅manifest，不提交权重 |

两个头文件体积约1.3MB，便于 MockEngine 离线构建。llama.cpp 原始源码包含额外工具和 tokenizer 测试词表，约99MB；与约469MiB模型权重均不纳入 Git。CPU 构建按根目录 README 准备固定源码；请保留上游许可证。

归档、头文件 SHA-256 和官方来源在 `dependencies.lock.json`。llama.cpp 的完整 MIT 许可副本保存在 `licenses/llama.cpp.LICENSE`；两个头文件对应许可位于各自目录。模型许可证参见[固定官方 LICENSE](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/9217f5db79a29953eb74d5343926648285ec7e67/LICENSE)。
