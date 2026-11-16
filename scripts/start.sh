#!/usr/bin/env bash
# LLMServe-Lite 本地推理服务 | scripts/start.sh
# 把选定的 mock 或 cpu 可执行文件与同名 JSON 配置配对启动。
# 最后的 exec 让服务接替脚本进程，因此终端发出的退出信号可以直接送达服务。
#
# 阅读地图（按源码顺序）
#   1 选运行模式 → 2 交接到服务进程


set -euo pipefail
cd "$(dirname "$0")/.."


# 模块 1：选运行模式
# 一次选择同时决定可执行文件和 JSON 配置的目录。选 cpu 时机器上要已有 CPU 版和对应权重。


preset="${1:-mock}"
case "$preset" in mock|cpu) ;; *) echo 'preset must be mock or cpu' >&2; exit 2;; esac


# 模块 2：交接到服务进程
# 配置文件跟着预设切换。exec 之后，当前进程就开始运行 llmserve。


# exec 让服务替换当前脚本进程，后续终止信号可以直接送给服务。
exec "build/$preset/llmserve" "configs/${preset}.json"
