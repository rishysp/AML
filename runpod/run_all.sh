#!/usr/bin/env bash
# One-command AML experiment run on a RunPod GPU pod (or any Linux GPU machine).
#
#   bash runpod/start.sh          # recommended: runs this script in the background with a log
#   bash runpod/run_all.sh        # or run it in the foreground
#
# What it does, in order (every step is resumable: just run it again after any interruption):
#   1. checks the machine (GPU, RAM, disk) and installs Python packages
#   2. downloads the IBM AML dataset from Kaggle (needs a Kaggle API token, see RUNPOD.md)
#   3. writes a run config
#   4. runs the pipeline (features, rules, XGBoost, durability checks)
#   5. runs the GNN experiment on the same windows and test accounts
#   6. zips the results into $WORKDIR/aml_results_<DATASET>.zip
#
# Settings (environment variables, all optional):
#   DATASET=HI-Medium | HI-Small     dataset to run                       (default HI-Medium)
#   WORKDIR=/workspace/aml_run       where data, checkpoints and results go (default /workspace/aml_run)
#   DATA_CSV=/path/file.csv          use a CSV already on disk instead of downloading
#   WORKERS=8                        parallel feature-building processes   (default: from RAM and cores)
#   QUICK=1                          fast smoke test (few trials/epochs): check the pod works first
#   AUTO_STOP=1                      stop this pod when finished (results on /workspace are kept)
#   MAX_TRAIN_WINDOWS=15             train on the N most recent windows only (memory/time safety valve)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-HI-Medium}"
WORKDIR="${WORKDIR:-/workspace/aml_run}"
QUICK="${QUICK:-0}"
AUTO_STOP="${AUTO_STOP:-0}"
mkdir -p "$WORKDIR/data"
cd "$REPO"

say() { printf '\n==== [%s] %s ====\n' "$(date '+%H:%M:%S')" "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
trap 'code=$?; if [ $code -ne 0 ]; then
  if [ $code -eq 137 ]; then echo "STOPPED: killed by the system (out of memory). Rerun to resume; consider WORKERS=4 or MAX_TRAIN_WINDOWS=15." >&2
  else echo "STOPPED with exit code $code at the step above. Rerun the same command to resume." >&2; fi; fi' EXIT

# ---------------------------------------------------------------- 1. machine check and packages
say "1/6 machine check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  DEVICE=cuda
else
  echo "no NVIDIA GPU found: running on CPU (slow for HI-Medium)"
  DEVICE=cpu
fi
DEVICE="${DEVICE_OVERRIDE:-$DEVICE}"
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1e6}' /proc/meminfo)
CORES=$(nproc)
DISK_GB=$(df -BG --output=avail "$WORKDIR" | tail -1 | tr -dc '0-9')
echo "RAM ${RAM_GB} GB | CPU cores ${CORES} | free disk at $WORKDIR: ${DISK_GB} GB | device ${DEVICE}"
if [ "$DATASET" = "HI-Medium" ] && [ "$RAM_GB" -lt 45 ]; then
  echo "WARNING: ${RAM_GB} GB RAM is tight for HI-Medium; 64 GB+ recommended. Setting WORKERS=1."
  WORKERS="${WORKERS:-1}"
fi
[ "${DISK_GB:-0}" -lt 25 ] && echo "WARNING: under 25 GB free at $WORKDIR; HI-Medium needs about 20 GB"
if [ -z "${WORKERS:-}" ]; then
  # ~5 GB per worker on HI-Medium-sized windows, 20 GB kept for the main process; at most 8
  WORKERS=$(( (RAM_GB - 20) / 5 )); [ "$WORKERS" -gt "$CORES" ] && WORKERS=$CORES
  [ "$WORKERS" -gt 8 ] && WORKERS=8; [ "$WORKERS" -lt 1 ] && WORKERS=1
fi
echo "feature-building workers: $WORKERS"

PY="${PYTHON:-python3}"
"$PY" -c "import sys; assert sys.version_info >= (3, 9), sys.version" || fail "Python 3.9+ needed"
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  "$PY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
  "$PY" -m pip install -q -r requirements.txt kaggle
  "$PY" -c "import torch" 2>/dev/null || "$PY" -m pip install -q torch
fi
"$PY" - <<'EOF'
import torch, xgboost, pandas
print(f"torch {torch.__version__} (CUDA {torch.cuda.is_available()}) | xgboost {xgboost.__version__} | pandas {pandas.__version__}")
EOF

# ---------------------------------------------------------------- 2. data
say "2/6 data"
CSV="${DATA_CSV:-$WORKDIR/data/${DATASET}_Trans.csv}"
if [ ! -s "$CSV" ]; then
  [ -n "${DATA_CSV:-}" ] && fail "DATA_CSV=$DATA_CSV not found"
  if [ -z "${KAGGLE_API_TOKEN:-}" ] && { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; } \
     && [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ ! -f "$HOME/.kaggle/access_token" ]; then
    fail "No Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY (or put kaggle.json in ~/.kaggle/). See RUNPOD.md step 3."
  fi
  echo "downloading ${DATASET}_Trans.csv from Kaggle..."
  kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml \
    -f "${DATASET}_Trans.csv" -p "$WORKDIR/data" --force
  for z in "$WORKDIR/data/${DATASET}_Trans.csv.zip" "$WORKDIR/data/${DATASET}_Trans.zip"; do
    if [ -f "$z" ]; then "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$z" "$WORKDIR/data" && rm -f "$z"; fi
  done
  [ -s "$CSV" ] || fail "download finished but $CSV is missing; check the file list with: kaggle datasets files ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
fi
ls -lh "$CSV"

# ---------------------------------------------------------------- 3. config
say "3/6 config"
OUT="$WORKDIR/reports_${DATASET}"
CONFIG="$WORKDIR/config_${DATASET}.yaml"
"$PY" - "$CONFIG" "$CSV" "$OUT" "$DATASET" "$DEVICE" "$WORKERS" "$QUICK" "${MAX_TRAIN_WINDOWS:-}" "$WORKDIR" <<'EOF'
import sys, yaml
config, csv, out, ds, device, workers, quick, max_tw, workdir = sys.argv[1:]
cfg = yaml.safe_load(open("config.yaml"))
cfg["data"]["raw_csv"] = csv
cfg["data"]["processed"] = f"{workdir}/{ds}.parquet"
cfg["split"]["n_windows"] = {"HI-Small": 5, "HI-Medium": 30}.get(ds, 5)   # ~1M transactions per window
cfg["split"]["workers"] = int(workers)
cfg["split"]["max_train_windows"] = int(max_tw) if max_tw else None
cfg["models"]["xgboost"]["device"] = device
cfg["models"]["search"]["max_rounds"] = 4000 if ds != "HI-Small" else 2000
if ds != "HI-Small":
    cfg["gnn"].update(seeds=[42, 7], max_epochs=150, patience=15)
if quick == "1":  # smoke test: proves the machine and code work, numbers are not meaningful
    cfg["models"]["search"].update(n_trials=2, max_rounds=300)
    cfg["models"]["stability_seeds"] = [7]
    cfg["models"]["bootstrap_reps"] = 50
    cfg["gnn"].update(seeds=[42], max_epochs=8, patience=3)
    out += "_quick"
cfg["outputs"] = {"reports_dir": out, "figures_dir": f"{out}/figures"}
yaml.safe_dump(cfg, open(config, "w"))
print(f"config: {config}\nreports: {out}\nwindows: {cfg['split']['n_windows']}, workers: {workers}, device: {device}, quick: {quick}")
EOF
[ "$QUICK" = "1" ] && OUT="${OUT}_quick"

# ---------------------------------------------------------------- 4. pipeline
say "4/6 pipeline (features, rules, XGBoost, durability checks)"
RESUME=()
[ -f "$OUT/xgb_search_trials.csv" ] && RESUME=(--reuse-search) && echo "resuming from earlier progress"
PYTHONUNBUFFERED=1 "$PY" -W ignore scripts/run_pipeline.py --config "$CONFIG" ${RESUME[@]+"${RESUME[@]}"}

# ---------------------------------------------------------------- 5. GNN
say "5/6 GNN experiment"
if [ -f "$OUT/gnn/results_gnn.json" ] && [ "$OUT/gnn/results_gnn.json" -nt "$OUT/test_scores.parquet" ] \
   && [ "${FORCE_GNN:-0}" != "1" ]; then
  echo "GNN results already up to date; skipping (FORCE_GNN=1 to rerun)"
else
  PYTHONUNBUFFERED=1 "$PY" -W ignore scripts/run_gnn.py --config "$CONFIG" --device auto
fi

# ---------------------------------------------------------------- 6. package results
say "6/6 packaging results"
SUFFIX=""; if [ "$QUICK" = "1" ]; then SUFFIX="_quick"; fi
ZIP="$WORKDIR/aml_results_${DATASET}${SUFFIX}.zip"
"$PY" - "$OUT" "$ZIP" <<'EOF'
import sys, zipfile, pathlib
out, zp = pathlib.Path(sys.argv[1]), sys.argv[2]
skip = {"cache", "stages"}  # large intermediate files; everything else (reports, charts, bullets) is kept
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as z:
    for f in out.rglob("*"):
        if f.is_file() and not (set(f.relative_to(out).parts) & skip):
            try:  # one unreadable file must not lose the whole results package
                z.write(f, f.relative_to(out))
            except Exception as e:
                print(f"skipped {f}: {e}")
print(f"results: {zp} ({pathlib.Path(zp).stat().st_size / 1e6:.1f} MB)")
EOF
cat "$OUT/model_comparison_test.csv" || true; echo
cat "$OUT/gnn/model_comparison_gnn.csv" || true; echo
cat "$OUT/gnn/resume_bullets_gnn.md" || true
say "FINISHED. Download $ZIP, then TERMINATE the pod (see RUNPOD.md, Part G)"

if [ "$AUTO_STOP" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
  echo "AUTO_STOP=1: stopping pod $RUNPOD_POD_ID in 60 s (results stay on /workspace). Ctrl+C to cancel."
  sleep 60 && runpodctl stop pod "$RUNPOD_POD_ID"
fi
