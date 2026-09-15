#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <input-media> <output-dir> [mt3|ismir2021]"
  exit 1
fi

INPUT_MEDIA="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
INPUT_DIR="$(dirname "${INPUT_MEDIA}")"
INPUT_BASENAME="$(basename "${INPUT_MEDIA}")"
OUTPUT_DIR_ABS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
MODEL_NAME="${3:-mt3}"

mkdir -p "${OUTPUT_DIR_ABS}" .docker-model-cache

docker build \
  --platform linux/amd64 \
  -f docker/mt3/Dockerfile \
  -t music-mt3-runner .

docker run --rm \
  --platform linux/amd64 \
  -e MT3_MODEL_NAME="${MODEL_NAME}" \
  -v "${INPUT_DIR}:/input-src:ro" \
  -v "${OUTPUT_DIR_ABS}:/output" \
  -v "$(pwd)/.docker-model-cache:/models" \
  music-mt3-runner \
  "/input-src/${INPUT_BASENAME}" \
  --output-dir "/output" \
  --model "${MODEL_NAME}"
