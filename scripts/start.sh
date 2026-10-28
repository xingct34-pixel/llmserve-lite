#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
preset="${1:-mock}"
case "$preset" in mock|cpu) ;; *) echo 'preset must be mock or cpu' >&2; exit 2;; esac
exec "build/$preset/llmserve" "configs/${preset}.json"
