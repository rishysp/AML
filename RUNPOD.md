# Running on a cloud GPU (RunPod)

For the larger IBM datasets (HI-Medium, ~32M transactions), one script runs everything on a rented GPU machine: `runpod/run_all.sh`. It installs packages, downloads the data from Kaggle, runs the pipeline and the GNN experiment, and zips the results. Every stage is checkpointed, so after any interruption the same command resumes. The scripts work on any Linux GPU machine; RunPod is just the setup used here.

**Reference run (HI-Medium):** RTX 3090 pod, about 1 hour end to end with 8 feature-building workers, peak RAM 11.5 GB, peak GPU memory 6.8 GB. Cost was well under $1 at community-cloud prices.

## 1. Choose a pod

What matters most is **system RAM**. The GPU needs only a few GB. Aim for **64 GB+ RAM**.

| Pod (Community Cloud) | VRAM | System RAM | vCPU | Notes |
|---|---|---|---|---|
| **RTX 3090** | 24 GB | ~125 GB | 16 | Best value |
| RTX 6000 Ada | 48 GB | ~167 GB | 10 | Alternative |
| L40S | 48 GB | ~94 GB | 16 | Alternative |
| RTX 4090 / A40 / A6000 / 5090 | 24-48 GB | 35-50 GB | 6-9 | Too little RAM for HI-Medium |

Specs and prices vary by host; check the RunPod console.

## 2. Deploy

1. **Pods → Deploy → Community Cloud** → choose a GPU from the table.
2. Template: **RunPod PyTorch** (any 2.x).
3. **Edit Template:**
   - Container disk 30 GB, volume 60 GB at `/workspace`.
   - Environment variables for the Kaggle download: either `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`. Get them from kaggle.com → Settings → API.
4. **On-Demand → Deploy.**

## 3. Run

**Connect → Jupyter Lab → File → New → Terminal**:

```bash
cd /workspace
git clone https://github.com/rishysp/AML.git && cd AML

QUICK=1 DATASET=HI-Small bash runpod/start.sh   # ~5-15 min smoke test: proves GPU, download, code
bash runpod/status.sh -f                        # Ctrl+C stops watching, not the run

bash runpod/start.sh                            # full HI-Medium run, in the background
bash runpod/status.sh                           # check any time: RUNNING / NOT RUNNING, progress, RAM, GPU
```

You can close the browser while it runs.
- **If it stops,** rerun `bash runpod/start.sh` and it resumes. `status.sh` shows the error lines.
- **Out of memory:** rerun with `WORKERS=4` or `MAX_TRAIN_WINDOWS=15`.
- **If a script fails with `$'\r': command not found`,** the files picked up Windows line endings. Fix them with `sed -i 's/\r$//' runpod/*.sh`.

## 4. Download and shut down

1. When `status.sh` shows `finished`, download `/workspace/aml_run/aml_results_HI-Medium.zip` with the Jupyter file browser (right-click → Download). Alternatively, use `runpodctl send <file>` on the pod and `runpodctl receive <code>` locally.
2. **Terminate** the pod with the trash icon. **Stop** only pauses the GPU; the volume keeps billing.
3. Check **Billing**: no pods running.

## Options

Set these before `bash runpod/start.sh`, for example `WORKERS=4 bash runpod/start.sh`.

| Variable | Default | Purpose |
|---|---|---|
| `DATASET` | `HI-Medium` | `HI-Small` for a quick check |
| `QUICK` | `0` | `1` = smoke test (few trials and epochs; numbers not meaningful) |
| `WORKERS` | chosen from RAM | Parallel feature-building processes |
| `MAX_TRAIN_WINDOWS` | all | Train on the N most recent windows only |
| `WORKDIR` | `/workspace/aml_run` | Data, checkpoints, logs and results |
| `DATA_CSV` | (download) | Use a CSV already on disk |
| `AUTO_STOP` | `0` | `1` = stop the pod when finished |
| `FORCE_GNN` | `0` | `1` = rerun the GNN even if its results are current |
