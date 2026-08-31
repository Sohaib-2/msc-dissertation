#!/usr/bin/env bash
# Launch the live runtime-KD-backdoor demo. Opens on http://127.0.0.1:8000
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/../../.venv/bin/python"
if [ ! -x "$VENV" ]; then
  echo "No .venv found at $VENV; falling back to system python3 (needs torch, torchvision, numpy and PIL)."
  VENV="python3"
fi
echo "Starting demo. The first load takes 2-3 minutes: a ResNet-50 on CPU has to load, measure its clean accuracy and calibrate the detector."
echo "When it prints 'demo ready', open http://127.0.0.1:8000"
exec "$VENV" "$HERE/app.py"
