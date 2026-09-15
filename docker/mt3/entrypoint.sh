#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MT3_MODEL_NAME:-mt3}"
CHECKPOINT_DIR="/models/checkpoints/${MODEL_NAME}"
SOUNDFONT_PATH="/models/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2"

mkdir -p /models/checkpoints

if [ ! -d "${CHECKPOINT_DIR}" ]; then
  echo "Downloading MT3 checkpoint: ${MODEL_NAME}"
  gsutil -m cp -r "gs://mt3/checkpoints/${MODEL_NAME}" /models/checkpoints
fi

if [ ! -f "${SOUNDFONT_PATH}" ]; then
  echo "Downloading soundfont"
  gsutil -m cp "gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2" /models/
fi

exec python /workspace/mt3_runner.py "$@"
