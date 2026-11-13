#!/usr/bin/env bash
# LLMServe-Lite 本地推理服务 | scripts/start.sh
# 把选定的 mock 或 cpu 可执行文件与同名 JSON 配置配对启动。
# 最后的 exec 让服务接替脚本进程，因此终端发出的退出信号可以直接送达服务。


set -euo pipefail
cd "$(dirname "$0")/.."
preset="${1:-mock}"
case "$preset" in mock|cpu) ;; *) echo 'preset must be mock or cpu' >&2; exit 2;; esac


# 模块：交接到服务进程
# 配置文件跟着预设切换。exec 之后，当前进程就开始运行 llmserve。


exec "build/$preset/llmserve" "configs/${preset}.json"
