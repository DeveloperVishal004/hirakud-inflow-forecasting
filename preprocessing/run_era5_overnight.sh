#!/bin/sh
# Unattended ERA5-Land download, safe to leave running overnight.
#
# Three things can stop a 6-hour job that a bare `nohup python ...` does not
# survive:
#
#   1. macOS idle sleep.  The sleep timer here is 1 minute; caffeinate holds
#      idle sleep off for as long as THIS script runs, and releases it the
#      moment the download finishes -- so the machine is not pinned awake
#      afterwards.  Note caffeinate cannot stop lid-close sleep: leave the lid
#      OPEN, and keep the charger connected.
#   2. A transient CDS failure.  The downloader records failures and exits
#      non-zero rather than pretending success, so this retries.  It skips
#      whatever already landed, so a retry costs nothing for completed work.
#   3. The terminal closing.  nohup/setsid detaches it.
#
# Retries are capped and spaced: if CDS is genuinely down, hammering it for
# hours is worse than stopping and leaving a clear log.
#
# Run:  nohup sh preprocessing/run_era5_overnight.sh > logs/era5_overnight.log 2>&1 &

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

# Pick an interpreter that actually has the dependencies, rather than trusting
# whatever `python3` resolves to.  This project's packages live in the 3.14
# framework build, but a pyenv shim can take over `python3` in a new shell --
# which silently broke an overnight run with ModuleNotFoundError after hours of
# progress.  Resolve it by capability, not by name.
PYTHON=""
for cand in \
    /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 \
    "$(command -v python3)" \
    "$(command -v python)"; do
    [ -x "$cand" ] || continue
    if "$cand" -c "import cdsapi" 2>/dev/null; then PYTHON="$cand"; break; fi
done
if [ -z "$PYTHON" ]; then
    echo "FATAL: no python found with cdsapi installed."
    echo "  install it with:  <your-python> -m pip install cdsapi"
    exit 1
fi
echo "using interpreter: $PYTHON"

MAX_ATTEMPTS=12
BACKOFF=300     # 5 min between attempts

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "=== attempt $attempt/$MAX_ATTEMPTS at $(date '+%Y-%m-%d %H:%M:%S') ==="
    caffeinate -i -s "$PYTHON" -u preprocessing/download_era5_land.py "$@"
    status=$?
    if [ $status -eq 0 ]; then
        echo "=== COMPLETE at $(date '+%Y-%m-%d %H:%M:%S') ==="
        "$PYTHON" preprocessing/download_era5_land.py --status "$@"
        exit 0
    fi
    echo "--- attempt $attempt exited $status; retrying in ${BACKOFF}s ---"
    "$PYTHON" preprocessing/download_era5_land.py --status "$@" | tail -5
    sleep $BACKOFF
done

echo "=== GAVE UP after $MAX_ATTEMPTS attempts at $(date '+%Y-%m-%d %H:%M:%S') ==="
"$PYTHON" preprocessing/download_era5_land.py --status "$@"
exit 1
