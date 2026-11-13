#!/usr/bin/env bash
# LLMServe-Lite 本地推理服务 | scripts/check.sh
# 根据给定的 CMake 预设配置、编译并运行 CTest。mock、cpu、sanitize 分别选用各自的构建目录。
# 三步各写一份日志，遇到问题时能区分是配置、编译还是测试失败。


set -euo pipefail
cd "$(dirname "$0")/.."
preset="${1:-mock}"
case "$preset" in mock|cpu|sanitize) ;; *) echo 'preset must be mock, cpu or sanitize' >&2; exit 2;; esac
mkdir -p results


# 模块：执行验收
# 先生成构建配置，再编译和运行已登记的测试；任一步失败都会使脚本报错。


cmake --preset "$preset" 2>&1 | tee "results/configure-${preset}.log"
cmake --build --preset "$preset" 2>&1 | tee "results/build-${preset}.log"
ctest --preset "$preset" 2>&1 | tee "results/test-${preset}.log"
