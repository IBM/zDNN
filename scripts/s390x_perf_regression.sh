#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--results-dir DIR] [--iterations N] [--thresholds-file FILE]

Run a deterministic s390x performance regression battery against selected
zDNN test drivers and fail when median runtime exceeds configured thresholds.

Options:
  --results-dir DIR       Output directory for perf logs/artifacts.
                          Default: ./artifacts/s390x-perf-<UTC timestamp>
  --iterations N          Number of timing iterations per benchmark (default: 5)
  --thresholds-file FILE  Shell file defining PERF_MAX_SEC_* thresholds.
                          Default: scripts/s390x_perf_thresholds.env
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR=""
ITERATIONS=5
THRESHOLDS_FILE="$ROOT_DIR/scripts/s390x_perf_thresholds.env"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --results-dir)
    RESULTS_DIR="$2"
    shift 2
    ;;
  --iterations)
    ITERATIONS="$2"
    shift 2
    ;;
  --thresholds-file)
    THRESHOLDS_FILE="$2"
    shift 2
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$ITERATIONS" -lt 1 ]]; then
  echo "--iterations must be a positive integer" >&2
  exit 2
fi

if [[ ! -f "$THRESHOLDS_FILE" ]]; then
  echo "Threshold file not found: $THRESHOLDS_FILE" >&2
  exit 2
fi

if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="$ROOT_DIR/artifacts/s390x-perf-$(date -u '+%Y%m%dT%H%M%SZ')"
fi
mkdir -p "$RESULTS_DIR"

SUMMARY="$RESULTS_DIR/perf_summary.txt"
RAW_CSV="$RESULTS_DIR/perf_raw.csv"
REPORT_MD="$RESULTS_DIR/perf_report.md"

# shellcheck disable=SC1090
source "$THRESHOLDS_FILE"

LD_RUNTIME_VAR="LD_LIBRARY_PATH"
S390X_SODIR="lib"

read_make_var() {
  local key="$1"
  local file="$2"
  awk -v key="$key" '
    $1 == key && ($2 == ":=" || $2 == "=") {
      $1 = "";
      $2 = "";
      sub(/^[[:space:]]+/, "", $0);
      gsub(/[[:space:]]+$/, "", $0);
      print $0;
      exit;
    }
  ' "$file"
}

if [[ -f "$ROOT_DIR/config.make" ]]; then
  maybe_ld_var="$(read_make_var LD_PATH_VAR "$ROOT_DIR/config.make")"
  maybe_sodir="$(read_make_var SODIR "$ROOT_DIR/config.make")"
  if [[ -n "$maybe_ld_var" ]]; then
    LD_RUNTIME_VAR="$maybe_ld_var"
  fi
  if [[ -n "$maybe_sodir" ]]; then
    S390X_SODIR="$maybe_sodir"
  fi
fi
LD_RUNTIME_VALUE="$ROOT_DIR/zdnn/$S390X_SODIR"

declare -a BENCHMARK_NAMES=(
  "matmul_op"
  "matmul_bcast_op"
  "quantized_matmul_op"
  "lstm_rnn"
)
declare -a BENCHMARK_CMDS=(
  "tests/bin/testDriver_zdnn_matmul_op"
  "tests/bin/testDriver_zdnn_matmul_bcast_op"
  "tests/bin/testDriver_zdnn_quantized_matmul_op"
  "tests/bin/testDriver_zdnn_lstm_rnn"
)
declare -a BENCHMARK_THRESHOLDS=(
  "${PERF_MAX_SEC_MATMUL_OP:-40.0}"
  "${PERF_MAX_SEC_MATMUL_BCAST_OP:-40.0}"
  "${PERF_MAX_SEC_QUANTIZED_MATMUL_OP:-60.0}"
  "${PERF_MAX_SEC_LSTM_RNN:-90.0}"
)

printf "benchmark,iteration,seconds,threshold_seconds\n" > "$RAW_CSV"

{
  echo "s390x performance regression run"
  echo "UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "Iterations: $ITERATIONS"
  echo "Thresholds file: $THRESHOLDS_FILE"
  echo "Runtime library path: ${LD_RUNTIME_VAR}=${LD_RUNTIME_VALUE}"
  echo ""
} > "$SUMMARY"

{
  echo "# s390x Performance Regression Report"
  echo ""
  echo "- UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "- Iterations: $ITERATIONS"
  echo "- Threshold file: \`$THRESHOLDS_FILE\`"
  echo ""
  echo "| Benchmark | Median (s) | Threshold (s) | Result |"
  echo "|---|---:|---:|---|"
} > "$REPORT_MD"

FAILURES=0

for idx in "${!BENCHMARK_NAMES[@]}"; do
  name="${BENCHMARK_NAMES[$idx]}"
  cmd="${BENCHMARK_CMDS[$idx]}"
  threshold="${BENCHMARK_THRESHOLDS[$idx]}"
  log_file="$RESULTS_DIR/perf_${name}.log"

  {
    echo "=== Benchmark: $name ==="
    echo "Command: $cmd"
    echo "Threshold (median sec): $threshold"
    echo "Iterations: $ITERATIONS"
  } | tee -a "$SUMMARY" > "$log_file"

  if [[ ! -x "$cmd" ]]; then
    echo "Missing benchmark binary: $cmd" | tee -a "$SUMMARY" >> "$log_file"
    echo "| \`$name\` | n/a | $threshold | FAIL (missing binary) |" >> "$REPORT_MD"
    FAILURES=$((FAILURES + 1))
    continue
  fi

  samples=()
  benchmark_failed=0
  for run_idx in $(seq 1 "$ITERATIONS"); do
    start_ns="$(python3 -c 'import time; print(time.perf_counter_ns())')"
    if env "$LD_RUNTIME_VAR=$LD_RUNTIME_VALUE" ZDNN_LOGLEVEL=off "$cmd" \
      >> "$log_file" 2>&1; then
      end_ns="$(python3 -c 'import time; print(time.perf_counter_ns())')"
      elapsed_sec="$(python3 - "$start_ns" "$end_ns" <<'PY'
import sys
start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])
print(f"{(end_ns - start_ns) / 1_000_000_000:.6f}")
PY
)"
      samples+=("$elapsed_sec")
      printf "%s,%d,%s,%s\n" "$name" "$run_idx" "$elapsed_sec" "$threshold" \
        >> "$RAW_CSV"
      echo "run ${run_idx}: ${elapsed_sec}s" | tee -a "$SUMMARY" >> "$log_file"
    else
      rc=$?
      echo "run ${run_idx}: command failed (rc=${rc})" | tee -a "$SUMMARY" \
        >> "$log_file"
      benchmark_failed=1
      break
    fi
  done

  if [[ "$benchmark_failed" -ne 0 ]]; then
    echo "| \`$name\` | n/a | $threshold | FAIL (command error) |" >> "$REPORT_MD"
    FAILURES=$((FAILURES + 1))
    continue
  fi

  median_sec="$(python3 - "${samples[@]}" <<'PY'
import statistics
import sys

vals = [float(arg) for arg in sys.argv[1:]]
print(f"{statistics.median(vals):.6f}")
PY
)"

  if python3 - "$median_sec" "$threshold" <<'PY'
import sys
median = float(sys.argv[1])
threshold = float(sys.argv[2])
sys.exit(0 if median <= threshold else 1)
PY
  then
    echo "median: ${median_sec}s (PASS)" | tee -a "$SUMMARY" >> "$log_file"
    echo "| \`$name\` | $median_sec | $threshold | PASS |" >> "$REPORT_MD"
  else
    echo "median: ${median_sec}s (FAIL threshold ${threshold}s)" | tee -a "$SUMMARY" \
      >> "$log_file"
    echo "| \`$name\` | $median_sec | $threshold | FAIL |" >> "$REPORT_MD"
    FAILURES=$((FAILURES + 1))
  fi

  echo "" | tee -a "$SUMMARY" >> "$log_file"
done

{
  echo ""
  echo "Raw results CSV: $RAW_CSV"
  echo "Markdown report: $REPORT_MD"
  echo "Failures: $FAILURES"
} | tee -a "$SUMMARY"

if [[ "$FAILURES" -ne 0 ]]; then
  exit 1
fi

exit 0
