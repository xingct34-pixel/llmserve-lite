#!/usr/bin/env bash
# LLMServe-Lite 本地推理服务 | scripts/check.sh
# 功能：根据给定的 CMake 预设配置、编译并运行 CTest。mock、cpu、sanitize 分别选用各自的构建目录。
# 三步各写一份日志，遇到问题时能区分是配置、编译还是测试失败。
# 目的：让本地与 CI 用同一预设运行同一套检查，并留下各阶段可单独定位的日志。


set -euo pipefail
cd "$(dirname "$0")/.."


# 模块：选预设
# 用同一个名字配置、编译、运行 CTest。测试结果因此对应这一次选定的后端与诊断选项。
preset="${1:-mock}"
case "$preset" in mock|cpu|sanitize) ;; *) echo 'preset must be mock, cpu or sanitize' >&2; exit 2;; esac
mkdir -p results


# 模块：执行验收
# 先生成构建配置，再编译和运行已登记的测试；任一步失败都会使脚本报错。
# 例如 CTest 报错时，先打开 test-预设名.log 看哪个用例失败；如果连这次的测试阶段都没进入，应先查同次运行的 configure 或 build 日志。
# tee 同时输出到屏幕和文件；pipefail 保留前面命令的失败，避免日志写成功就误报验收通过。
cmake --preset "$preset" 2>&1 | tee "results/configure-${preset}.log"
cmake --build --preset "$preset" 2>&1 | tee "results/build-${preset}.log"
ctest --preset "$preset" 2>&1 | tee "results/test-${preset}.log"
