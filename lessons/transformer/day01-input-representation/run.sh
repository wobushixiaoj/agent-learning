#!/bin/zsh

set -euo pipefail

LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/opt/anaconda3/bin/python3"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON"
  exit 1
fi

exec "$PYTHON" "$LAB_DIR/lesson.py"
