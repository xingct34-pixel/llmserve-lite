#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
preset="${1:-mock}"
case "$preset" in mock|cpu|sanitize) ;; *) echo 'preset must be mock, cpu or sanitize' >&2; exit 2;; esac
mkdir -p results
cmake --preset "$preset" 2>&1 | tee "results/configure-${preset}.log"
cmake --build --preset "$preset" 2>&1 | tee "results/build-${preset}.log"
ctest --preset "$preset" 2>&1 | tee "results/test-${preset}.log"
