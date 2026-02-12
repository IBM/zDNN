#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") <user@host> <remote_dir> [--jobs N] [--run-id ID] [--no-delete] [--no-autoreconf] [--skip-distclean] [--perf-regression] [--perf-iterations N] [--perf-thresholds FILE]

Rsync the current repo to an s390x host, run the remote build/test battery,
and sync result artifacts back locally.

Arguments:
  user@host      SSH target
  remote_dir     Remote path where the repo copy will be staged

Options:
  --jobs N       Parallel make jobs for remote battery script
  --run-id ID    Explicit artifact run id (default: UTC timestamp)
  --no-delete    Disable rsync --delete during sync
  --no-autoreconf   Pass --no-autoreconf to remote battery script
  --skip-distclean  Pass --skip-distclean to remote battery script
  --perf-regression  Enable remote perf regression battery
  --perf-iterations N  Iterations per perf benchmark
  --perf-thresholds FILE  Threshold file path on remote host
USAGE
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

TARGET_HOST="$1"
REMOTE_DIR="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXCLUDE_FILE="$ROOT_DIR/scripts/rsync-zdnn.exclude"
if [[ ! -f "$EXCLUDE_FILE" ]]; then
  echo "Exclude file not found: $EXCLUDE_FILE" >&2
  exit 2
fi

RUN_ID="s390x-battery-$(date -u '+%Y%m%dT%H%M%SZ')"
JOBS=""
RSYNC_DELETE=1
REMOTE_NO_AUTORECONF=0
REMOTE_SKIP_DISTCLEAN=0
REMOTE_PERF_REGRESSION=0
REMOTE_PERF_ITERATIONS=""
REMOTE_PERF_THRESHOLDS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
  --jobs)
    JOBS="$2"
    shift 2
    ;;
  --run-id)
    RUN_ID="$2"
    shift 2
    ;;
  --no-delete)
    RSYNC_DELETE=0
    shift
    ;;
  --no-autoreconf)
    REMOTE_NO_AUTORECONF=1
    shift
    ;;
  --skip-distclean)
    REMOTE_SKIP_DISTCLEAN=1
    shift
    ;;
  --perf-regression)
    REMOTE_PERF_REGRESSION=1
    shift
    ;;
  --perf-iterations)
    REMOTE_PERF_ITERATIONS="$2"
    shift 2
    ;;
  --perf-thresholds)
    REMOTE_PERF_THRESHOLDS="$2"
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

if [[ -n "$JOBS" ]] && { ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; }; then
  echo "--jobs must be a positive integer" >&2
  exit 2
fi

if [[ -n "$REMOTE_PERF_ITERATIONS" ]] &&
  { ! [[ "$REMOTE_PERF_ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$REMOTE_PERF_ITERATIONS" -lt 1 ]]; }; then
  echo "--perf-iterations must be a positive integer" >&2
  exit 2
fi

REMOTE_RESULTS_DIR="$REMOTE_DIR/artifacts/$RUN_ID"
LOCAL_RESULTS_ROOT="$ROOT_DIR/artifacts"
LOCAL_RESULTS_DIR="$LOCAL_RESULTS_ROOT/$RUN_ID"
mkdir -p "$LOCAL_RESULTS_ROOT"

RSYNC_FLAGS=(-az --exclude-from="$EXCLUDE_FILE")
if [[ "$RSYNC_DELETE" -eq 1 ]]; then
  RSYNC_FLAGS+=(--delete)
fi

echo "[1/4] Ensuring remote directory exists: $TARGET_HOST:$REMOTE_DIR"
ssh "$TARGET_HOST" "mkdir -p '$REMOTE_DIR'"

echo "[2/4] Rsyncing source to remote"
rsync "${RSYNC_FLAGS[@]}" "$ROOT_DIR/" "$TARGET_HOST:$REMOTE_DIR/"

echo "[3/4] Running remote battery"
REMOTE_RC=0
REMOTE_CMD="cd '$REMOTE_DIR' && chmod +x scripts/s390x_run_battery.sh scripts/s390x_perf_regression.sh && scripts/s390x_run_battery.sh --results-dir '$REMOTE_RESULTS_DIR'"
if [[ -n "$JOBS" ]]; then
  REMOTE_CMD+=" --jobs '$JOBS'"
fi
if [[ "$REMOTE_NO_AUTORECONF" -eq 1 ]]; then
  REMOTE_CMD+=" --no-autoreconf"
fi
if [[ "$REMOTE_SKIP_DISTCLEAN" -eq 1 ]]; then
  REMOTE_CMD+=" --skip-distclean"
fi
if [[ "$REMOTE_PERF_REGRESSION" -eq 1 ]]; then
  REMOTE_CMD+=" --perf-regression"
fi
if [[ -n "$REMOTE_PERF_ITERATIONS" ]]; then
  REMOTE_CMD+=" --perf-iterations '$REMOTE_PERF_ITERATIONS'"
fi
if [[ -n "$REMOTE_PERF_THRESHOLDS" ]]; then
  REMOTE_CMD+=" --perf-thresholds '$REMOTE_PERF_THRESHOLDS'"
fi
if ssh "$TARGET_HOST" "$REMOTE_CMD"; then
  REMOTE_RC=0
else
  REMOTE_RC=$?
fi

echo "[4/4] Syncing remote results back to local artifacts"
mkdir -p "$LOCAL_RESULTS_DIR"
rsync -az "$TARGET_HOST:$REMOTE_RESULTS_DIR/" "$LOCAL_RESULTS_DIR/" || true

echo "Local results: $LOCAL_RESULTS_DIR"

exit "$REMOTE_RC"
