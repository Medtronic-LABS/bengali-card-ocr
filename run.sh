#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing local environment. Complete README setup before running offline." >&2
  exit 1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
exec .venv/bin/python -m streamlit run app.py "$@"
