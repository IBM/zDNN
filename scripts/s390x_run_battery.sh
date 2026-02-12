#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--results-dir DIR] [--jobs N] [--no-autoreconf] [--skip-distclean] [--perf-regression] [--perf-iterations N] [--perf-thresholds FILE]

Run a repeatable zDNN build+test battery on an s390x host and write all logs
under an artifacts directory.

Options:
  --results-dir DIR   Output directory for logs/artifacts.
                      Default: ./artifacts/s390x-battery-<UTC timestamp>
  --jobs N            Parallel jobs for make. Default: detected CPU count.
  --no-autoreconf     Skip autoreconf step.
  --skip-distclean    Skip make distclean before configure.
  --perf-regression   Run perf regression battery after correctness tests.
  --perf-iterations N Iterations per perf benchmark.
  --perf-thresholds FILE
                      Threshold file for perf regression script.
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 4)"
DO_AUTORECONF=1
DO_DISTCLEAN=1
DO_PERF_REGRESSION=0
PERF_ITERATIONS=""
PERF_THRESHOLDS=""
RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
  --results-dir)
    RESULTS_DIR="$2"
    shift 2
    ;;
  --jobs)
    JOBS="$2"
    shift 2
    ;;
  --no-autoreconf)
    DO_AUTORECONF=0
    shift
    ;;
  --skip-distclean)
    DO_DISTCLEAN=0
    shift
    ;;
  --perf-regression)
    DO_PERF_REGRESSION=1
    shift
    ;;
  --perf-iterations)
    PERF_ITERATIONS="$2"
    shift 2
    ;;
  --perf-thresholds)
    PERF_THRESHOLDS="$2"
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

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 2
fi

if [[ -n "$PERF_ITERATIONS" ]] &&
  { ! [[ "$PERF_ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$PERF_ITERATIONS" -lt 1 ]]; }; then
  echo "--perf-iterations must be a positive integer" >&2
  exit 2
fi

if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="$ROOT_DIR/artifacts/s390x-battery-$(date -u '+%Y%m%dT%H%M%SZ')"
fi

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.txt"
FAILURES=0

run_step() {
  local step="$1"
  shift
  local log="$RESULTS_DIR/${step}.log"

  {
    echo ""
    echo "=== STEP: ${step} ==="
    echo "CMD: $*"
    echo "UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } | tee -a "$SUMMARY" > "$log"

  if "$@" >> "$log" 2>&1; then
    echo "RESULT: PASS (${step})" | tee -a "$SUMMARY"
  else
    local rc=$?
    echo "RESULT: FAIL (${step}) rc=${rc}" | tee -a "$SUMMARY"
    FAILURES=$((FAILURES + 1))
  fi
}

run_step env_info bash -lc '
  set -e
  echo "PWD: $(pwd)"
  uname -a
  echo
  lscpu || true
  echo
  gcc --version || true
  g++ --version || true
  make --version | head -n 1 || true
  autoreconf --version | head -n 1 || true
  python3 --version || true
  git rev-parse --short HEAD || true
  git status -sb || true
'

if [[ "$DO_DISTCLEAN" -eq 1 ]]; then
  run_step distclean bash -lc '
    if [[ -f config.make ]]; then
      make distclean
    else
      echo "config.make not present; skipping distclean"
    fi
  '
fi

if [[ "$DO_AUTORECONF" -eq 1 ]]; then
  run_step autoreconf autoreconf .
fi

run_step configure ./configure
run_step build make -j"$JOBS" all

# Battery run #1: baseline runtime env.
run_step test_default make -j"$JOBS" test

# Battery run #2: runtime precheck enabled.
run_step test_precheck env ZDNN_ENABLE_PRECHECK=true make -j"$JOBS" test

# Battery run #3: precheck + full status diagnostics.
run_step test_precheck_statusdiag env ZDNN_ENABLE_PRECHECK=true ZDNN_STATUS_DIAG=0xFFFFFFFF make -j"$JOBS" test

if [[ "$DO_PERF_REGRESSION" -eq 1 ]]; then
  PERF_CMD=(scripts/s390x_perf_regression.sh --results-dir "$RESULTS_DIR/perf")
  if [[ -n "$PERF_ITERATIONS" ]]; then
    PERF_CMD+=(--iterations "$PERF_ITERATIONS")
  fi
  if [[ -n "$PERF_THRESHOLDS" ]]; then
    PERF_CMD+=(--thresholds-file "$PERF_THRESHOLDS")
  fi

  run_step perf_regression "${PERF_CMD[@]}"
fi

run_step collect_artifacts bash -lc '
  mkdir -p "'$RESULTS_DIR'/config"
  cp -f config.log config.status config.make config.h "'$RESULTS_DIR'/config" 2>/dev/null || true

  if compgen -G "tests/bin/*.txt" > /dev/null; then
    tar -czf "'$RESULTS_DIR'/test-bin-logs.tgz" tests/bin/*.txt
    ls -1 tests/bin/*.txt > "'$RESULTS_DIR'/test-bin-files.txt"
  else
    echo "No tests/bin/*.txt files found" > "'$RESULTS_DIR'/test-bin-files.txt"
  fi

  if [[ -d "'$RESULTS_DIR'/perf" ]]; then
    tar -czf "'$RESULTS_DIR'/perf-logs.tgz" "'$RESULTS_DIR'/perf"/*.log 2>/dev/null || true
  fi
'

{
  echo ""
  echo "Artifacts: $RESULTS_DIR"
  echo "Failures: $FAILURES"
} | tee -a "$SUMMARY"

if [[ "$FAILURES" -ne 0 ]]; then
  exit 1
fi

exit 0
