#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

xcodebuild \
  -scheme ModelBench \
  -destination 'platform=macOS' \
  -configuration Debug \
  build >/tmp/model-bench-xcodebuild.log 2>&1

shopt -s nullglob
candidates=("$HOME"/Library/Developer/Xcode/DerivedData/swift-benchmarks-*/Build/Products/Debug/model-bench)

if [[ ${#candidates[@]} -eq 0 ]]; then
  echo "ERROR: could not locate model-bench under DerivedData" >&2
  echo "xcodebuild log: /tmp/model-bench-xcodebuild.log" >&2
  exit 1
fi

IFS=$'\n' sorted=($(ls -t "${candidates[@]}"))
unset IFS
BIN="${sorted[0]}"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: model-bench binary not found at $BIN" >&2
  echo "xcodebuild log: /tmp/model-bench-xcodebuild.log" >&2
  exit 1
fi

exec "$BIN" "$@"
