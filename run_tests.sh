#!/usr/bin/env bash
# Run the full Aria test suite. Uses the `fake` engine — no model weights,
# no GPU, no network required.
set -uo pipefail
cd "$(dirname "$0")/python-sidecar"

# use the venv python if present, else system python3
PY="python3"
[[ -x .venv/bin/python ]] && PY=".venv/bin/python"

total=0; passed=0; failed=0
for t in tests/test_*.py; do
  echo "=== $t ==="
  if "$PY" "$t"; then
    passed=$((passed+1))
  else
    failed=$((failed+1))
  fi
  total=$((total+1))
  echo
done

echo "──────────────────────────────"
echo "suites: $passed/$total passed"
[[ $failed -eq 0 ]] && echo "✓ all green" || { echo "✗ $failed suite(s) failed"; exit 1; }
