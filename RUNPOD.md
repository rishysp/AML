# Running the HI-Medium experiment on RunPod: step by step

**You do not need GitHub, and you do not need to create any folders.** Everything the pod needs is in one file, `aml_code_for_runpod.zip` (62 KB). You upload it to the pod in your browser, and the pod downloads the 3 GB dataset from Kaggle itself.

**What you need:** a Kaggle account, a RunPod account with about $10 of credit, and about 20 minutes of hands-on time. The run itself takes 3.5-5 hours and you don't need to watch it.

**Checklist**
- [ ] A. Find the code zip on your laptop
- [ ] B. Get a Kaggle API token
- [ ] C. Create a RunPod account and add credit
- [ ] D. Deploy the pod
- [ ] E. Upload the zip and start a 15-minute test run
- [ ] F. Start the real run and check on it
- [ ] G. Download the results, then terminate the pod
- [ ] H. (Optional, later) Update your GitHub repo

---

## A. Find the code zip (1 min)

It's already on your laptop:

```
C:\Users\rishy\OneDrive\Documents\aml-transaction-monitoring\aml_code_for_runpod.zip
```

Don't unzip it. You'll upload it as is in step E.

## B. Get a Kaggle API token (3 min)

The pod downloads the IBM dataset from Kaggle, which needs your API credentials.

1. Go to **kaggle.com**, sign in, click your profile picture, then **Settings**.
2. Scroll to the **API** section. Kaggle offers one of two kinds of token:
   - **"Create New Token"** shows a single token string. Copy it right away; it's shown once. You'll use it as `KAGGLE_API_TOKEN`.
   - **"Create Legacy API Key"** (or an older "Create New Token") downloads `kaggle.json`. Open it in Notepad. It looks like `{"username":"rishysp","key":"abc123..."}`. You'll use these as `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Keep the token private, like a password. Don't paste it into chats or into GitHub.

## C. RunPod account and credit (5 min)

1. Go to **runpod.io** and sign up.
2. Open **Billing** in the left menu and add **$10**. This run costs about $1-2.

## D. Deploy the pod (5 min)

1. In the left menu, click **Pods**, then **+ Deploy**.
2. At the top, select **Community Cloud**. It's cheaper, and fine for this job.
3. Pick **RTX 3090**. Before choosing, check that its card says about **125 GB RAM**.
   - If no RTX 3090 is available, pick **RTX 6000 Ada** (about 167 GB RAM) or **L40S** (about 94 GB RAM).
   - **Don't** pick RTX 4090, A40, RTX A6000 or RTX 5090: they have too little RAM (35-50 GB), and RAM is what ran out on Kaggle.
4. **Template:** choose **RunPod PyTorch** (any 2.x version). It's usually the default.
5. Click **Edit Template** (or **Customize Deployment**) and set:
   - **Container Disk:** `30` GB
   - **Volume Disk:** `60` GB, with mount path `/workspace` (the default)
   - **Environment Variables:** click **Add**, then enter
     - either `KAGGLE_API_TOKEN` = your token (new-style token), or
     - two entries: `KAGGLE_USERNAME` = your username, and `KAGGLE_KEY` = your key (from `kaggle.json`)
   - Click **Set Overrides** or **Save**.
6. Under pricing, choose **On-Demand**, not Spot. Spot pods can be interrupted.
7. Click **Deploy On-Demand**. After about 1-3 minutes the pod shows **Running**.

## E. Upload the zip and run a 15-minute test (20 min, mostly waiting)

1. On your pod's card, click **Connect**, then **Connect to Jupyter Lab** (HTTP port 8888). Jupyter Lab opens in a new browser tab.
   - If it says "not ready", wait a minute and try again.
2. **Upload the zip:** in Jupyter's left-hand file browser, click the **up-arrow (Upload)** icon and choose `aml_code_for_runpod.zip` from step A. Wait until it appears in the file list.
3. **Open a terminal:** **File → New → Terminal**.
4. Copy and paste these lines into the terminal, then press Enter:

   ```bash
   cd /workspace
   ls *.zip
   python3 -m zipfile -e aml_code_for_runpod.zip /workspace
   cd /workspace/aml
   ls
   ```

   You should see `README.md  RUNPOD.md  config.yaml  runpod  scripts  src ...`.
   - If `ls *.zip` says "No such file", the upload went to a different folder. Find it with `find / -name aml_code_for_runpod.zip 2>/dev/null`, then use that path in the `zipfile` line.
5. **Start the test run.** It downloads the small HI-Small dataset, about 0.5 GB, and runs everything quickly:

   ```bash
   QUICK=1 DATASET=HI-Small bash runpod/start.sh
   bash runpod/status.sh -f
   ```

   Progress lines scroll by. The test is done when you see:

   ```
   ==== [..] FINISHED. Download /workspace/aml_run/aml_results_HI-Small_quick.zip ...
   ```

   Press **Ctrl+C** to stop watching. This only stops the display, not the run.
   - If you see `ERROR: No Kaggle credentials`, the environment variables from D5 didn't take effect. Set them in this terminal instead (use your own values), then rerun the `start.sh` line:
     ```bash
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_key
     ```
     For a new-style token, use `export KAGGLE_API_TOKEN=your_token` instead.
   - If the Kaggle download shows `401` or `403`, the token is wrong or expired. Create a new one (step B).
   - For any other error, copy the last ~20 lines and send them to me.

   The numbers from this test mean nothing. It only proves that the pod, the GPU, the download and the code all work.

## F. Start the real run (2 min, then 3.5-5 hours unattended)

In the same terminal:

```bash
cd /workspace/aml
bash runpod/start.sh
bash runpod/status.sh
```

- `status.sh` should say **RUNNING**. You can now close the browser and turn off your laptop. The run continues on the pod.
- **To check on it later:** on runpod.io, go to **Pods → Connect → Jupyter Lab → File → New → Terminal**, then run:
  ```bash
  cd /workspace/aml && bash runpod/status.sh
  ```
  It shows RUNNING or NOT RUNNING, the latest progress lines, and RAM, GPU and disk use.
- **Roughly what you'll see:** data download and load (~10 min), features with several workers (~15-30 min), 10 XGBoost search trials (~30-45 min), final models and checks (~1-1.5 h), then the GNN (~30-60 min).
- **If it says NOT RUNNING without FINISHED**, run `bash runpod/start.sh` again. It resumes from the last finished step.
  - If the message mentions **out of memory**, rerun with fewer workers: `WORKERS=4 bash runpod/start.sh`.

## G. Download the results, then terminate the pod (5 min)

1. When `status.sh` shows `finished: /workspace/aml_run/aml_results_HI-Medium.zip`, go to the Jupyter file browser.
2. Open the **`aml_run`** folder, right-click **`aml_results_HI-Medium.zip`**, and choose **Download**.
3. On your laptop, open the zip and check that it contains `model_comparison_test.csv`, a `gnn` folder and `figures`.
4. Back on runpod.io, go to **Pods** and, on your pod, click the **trash icon (Terminate)**, then confirm.
   - **Terminate** deletes the pod and its files, and all billing stops. Do this once you have the zip.
   - **Stop** (square icon) only pauses the GPU. The 60 GB volume keeps costing about $0.40/day until you terminate.
5. Check **Billing**: no pods should be running.
6. Send me the results zip, and I'll update the resume and white paper with the GNN results.

## H. (Optional, later) Update your GitHub repo

Not needed for the run. It just keeps your public portfolio repo current. On **github.com/rishysp/AML**:

1. Click **Add file → Upload files**.
2. From `C:\Users\rishy\OneDrive\Documents\aml-transaction-monitoring\aml-transaction-monitoring\`, drag in the folders `runpod`, `scripts`, `src`, `kaggle` and `docs`, and the files `README.md`, `RUNPOD.md`, `config.yaml` and `.gitattributes`.
   - **Don't upload** `data` (GitHub rejects files over 100 MB) or `reports\cache`.
3. Enter a commit message such as "Add RunPod scripts and memory fixes", then click **Commit changes**.

---

### Reference: run options

Put these before `bash runpod/start.sh`, for example `WORKERS=4 bash runpod/start.sh`.

| Variable | Default | Purpose |
|---|---|---|
| `DATASET` | `HI-Medium` | `HI-Small` for a quick check |
| `QUICK` | `0` | `1` = smoke test (few trials and epochs; numbers not meaningful) |
| `WORKERS` | chosen from RAM | Parallel feature-building processes; lower it if memory runs out |
| `MAX_TRAIN_WINDOWS` | all | Train on the N most recent windows only (memory and time safety valve) |
| `WORKDIR` | `/workspace/aml_run` | Data, checkpoints, logs and results |
| `AUTO_STOP` | `0` | `1` = stop the pod when finished (you still terminate it after downloading) |
| `FORCE_GNN` | `0` | `1` = rerun the GNN even if its results are current |
