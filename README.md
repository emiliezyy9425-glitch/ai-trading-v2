# AI Trading Agent

This repository contains an automated trading agent with multiple machine learning models.
Each model is trained as a binary classifier, learning from historical data to
predict whether the next price movement will rise or fall so the agent can make
appropriate buy or sell decisions.

It requires Python 3.11 or later.

The agent monitors broader market breadth using the S&P 500 percentage-above-20-day moving average. When an Interactive Brokers
connection is unavailable, this metric now falls back to the official TradingView indicator `SPXA20R`.

## AI Agent Workflow

See `docs/workflow_diagram.md` for a visual end-to-end workflow diagram that connects the data, training, inference, and trading components.

1. **Collect market data** – fetch recent bars directly from Interactive Brokers for the watched equities.
2. **Compute indicators** – derive RSI, MACD, Bollinger Bands, Fibonacci levels, volume signals, and other features via `indicators.py` for 1h/4h/1d windows.
3. **Generate model inputs** – align the live features to the `FEATURE_NAMES` schema used during training.
4. **Predict with multiple models** – `ml_predictor.predict_with_all_models` now reuses globally loaded Transformer and TCN checkpoints (plus Random Forest, XGBoost, LightGBM, LSTM, and PPO). Use `ml_predictor.predict_with_all_models_for_tickers` to score several tickers in one call while reusing the cached model instances.
5. **Ultimate decision** – `ml_predictor.independent_model_decisions` now prioritizes a triple-stack nuclear vote: when LSTM (≥0.96), Transformer (≥0.98), and TCN (≥0.92) all agree, the "TRIPLE_NUCLEAR" signal executes immediately. Otherwise the agent falls back to DeepSeq (LSTM+Transformer) alignment, then seeks a unanimous "TRIPLE_TREE_NUCLEAR" vote from RandomForest/XGBoost/LightGBM, then a confident RandomForest solo read, and only then uses PPO as a rare tie-breaker.
6. **Execute trades** – `tsla_ai_master_final_ready.py` submits orders through Interactive Brokers, applying position sizing, stop orders, and risk checks.
7. **Log results** – each executed trade is appended to `data/trade_log.csv` alongside the feature snapshot and metadata for later analysis.
8. **Self-learn and retrain** – `self_learn.py` labels past trades with future returns and trains updated models in `models/`. `auto_train.py` can watch `data/historical_data.csv` and trigger retraining when new data arrives.

### Real-world schedule (what pros use)

The live deployment follows the same cadence used by discretionary desks that rebalance around the New York close while operating from Hong Kong (HKT/UTC+8):

| Time (HKT) | Action |
| --- | --- |
| 21:00–22:00 | U.S. session hourly bar forms. |
| 22:00 | As soon as the 21:00–22:00 bar closes, run the prediction pipeline and send the market/limit order suggested by the leading model. |
| 22:05 | Confirm the broker fill and update the trade log. |
| 04:00 | When the next U.S. hourly bar completes (03:00–04:00 UTC), repeat the process for the new signal. |

Adjust the cron jobs (`hourly_cronjob.txt`, `weekly_cronjob.txt`, etc.) if you need to shift the operating window for a different exchange or timezone.

### Live trading rule

**Models and thresholds**

The live loop listens to three core models:

* **LSTM:** confidence ≥ **0.50**
* **LightGBM:** confidence ≥ **0.85**
* **Transformer:** confidence ≥ **0.80**

Enter a trade when **any one** of these models produces a BUY/SELL signal above its threshold. If multiple models qualify in the same tick, the highest-confidence vote wins.

**Exit logic**

Open positions exit at the earliest of these triggers:

* **Take profit:** gain of roughly **+17%** (`tp = 0.17`).
* **Stop loss:** drawdown of roughly **−4%** (`sl = 0.04`).
* **Max hold:** position open for **96 hours** (`max_hold = 96.0`), recorded as a `max_96h` forced exit.

## Installation

Clone the repository and install dependencies in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Candlestick patterns:** The project now relies solely on the built-in
> heuristic detectors defined in `indicators.py` for the ten high-signal
> candlestick formations (hammer, engulfing, morning star, etc.). No external
> TA-Lib dependency is required. If you prefer to experiment with the official
> TA-Lib implementations you can still install the library manually before
> starting the agent, but it is entirely optional.

> **Transformer dependency:** The optional Transformer classifier depends on
> the Hugging Face `transformers` package. Install it via `pip install -r
> requirements.txt` (recommended) or `pip install transformers` before running
> `retrain_models.py` if you plan to train the Transformer checkpoint.

## Generating Historical Indicator Data

Training scripts such as `retrain_models.py` expect `data/historical_data.csv` with pre-computed technical indicators. If you have downloaded raw historical bars, build this file offline:

```bash
python scripts/compute_historical_indicators.py --input-dir data/raw --output-file data/historical_data.csv
```

The script reads per-ticker CSV files from `data/raw/` (one file named `<TICKER>.csv` for each symbol listed in `data/tickers.txt`). Each file should contain columns `timestamp`, `open`, `high`, `low`, `close`, and `volume` with an hourly frequency. Indicators (RSI, MACD, EMA10, Bollinger Bands, ATR, etc.) are computed for 1h, 4h, and 1d timeframes and the result is written to `data/historical_data.csv` for downstream training.

> **Historical coverage:** The download, indicator, and aggregation scripts now share a
> single `TRAINING_LOOKBACK_YEARS` resolver (default `10`) so the agent always builds
> at least a decade of hourly history before training. Override the environment
> variable if you need a different span; every workflow (`scripts/download_historical_prices.py`,
> `scripts/compute_historical_indicators.py`, `scripts/generate_historical_data.py`, and
> `tsla_ai_master_final_ready.py update-historical`) will automatically adopt it.

For quick experiments `self_learn.py` can bootstrap a minimal `historical_data.csv` directly from the existing trade log. By default the script will generate this file if it is missing:

```bash
python self_learn.py
```

Use `--no-generate-historical` to skip this bootstrapping step. To train from a backtest run, point the script at the backtest log:

```bash
python self_learn.py --trade-log data/trade_log_backtest.csv
```

### RSI Calculation Note

The RSI calculation reuses the last valid RSI value when the loss component is zero. If no prior value exists (e.g., during an initial uninterrupted uptrend), the function returns `None` instead of a neutral placeholder.


## Training Models

To retrain all models using historical data, run:

```bash
python scripts/run_training_pipeline.py
```

### Running inside Docker

The project now extends NVIDIA's pre-optimized TensorFlow container (`nvcr.io/nvidia/tensorflow:25.09-tf2-py3`), which already bundles CUDA 13.0, cuDNN 9.8, and TensorFlow 2.20.0+ with Blackwell (`sm_120`) kernels. The provided `Dockerfile` only needs to layer the trading application dependencies on top. It also exports `TF_CUDA_COMPUTE_CAPABILITIES=12.0` during the build so TensorFlow eagerly compiles `sm_120` kernels and avoids `CUDA_ERROR_INVALID_PTX` on RTX 5090-class GPUs. If you modify `requirements.txt`, rebuild the image to ensure the training pipeline has everything it needs:

```bash
docker compose build --no-cache
docker run --rm tsla-ai-trading-agent python -m scripts.run_training_pipeline
```

The script first looks for `data/historical_data.csv`. If the file exists, it is cleaned and used directly for training. Otherwise, it rebuilds the dataset from locally cached raw bars (or new IBKR downloads), computes technical indicators, and saves `data/historical_data.csv` before training. The full workflow is:

1. **Use or create `data/historical_data.csv`:** use an existing file or download raw bars and compute indicators if missing.
2. **Clean and validate** the historical dataset.
3. **Retrain models** (Random Forest, XGBoost, LightGBM, LSTM, PPO) using the resulting dataset.

### Model artifacts in version control

To keep the repository size manageable and avoid redistributing licensed market
data, only a lightweight subset of trained weights is checked in
(`updated_lightgbm.pkl`, `updated_xgboost.pkl`, and `updated_lstm.keras`). The
Random Forest, PPO, and Transformer checkpoints are much larger and embed
training traces that should be regenerated locally. When these files are missing
the predictor logs a clear message and returns an error payload for the affected
model instead of forcing a default vote.

Regenerate the full set of artifacts by running the training pipeline described
above. Fresh weights will be written to the `models/` directory using the
filenames expected by `ml_predictor.predict_with_all_models` (for example,
`models/updated_random_forest.pkl` and `models/updated_ppo.zip`).

### Enabling GPU acceleration in Docker

TensorFlow automatically probes for NVIDIA GPUs during startup. When the container
is launched without GPU support the logs show warnings similar to
`Failed call to cuInit: UNKNOWN ERROR (303)` before falling back to CPU
execution. To leverage GPU acceleration you must install the NVIDIA Container
Toolkit on the host and pass the GPU through to Docker.

If you are running purely on CPU hardware the repository now ships with a
lightweight `nvidia-smi` stub in `scripts/`. The launch script adds this
directory to the `PATH` so TensorFlow can detect the binary without triggering
CUDA initialisation errors. When you eventually provision a real NVIDIA driver,
the genuine `nvidia-smi` on the host will override the stub automatically.

1. **Install the toolkit** (one-time): follow the official
   [NVIDIA Container Toolkit instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   for your Linux distribution or WSL2 environment.
2. **Configure Docker** once the toolkit is installed:

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

3. **Update the host driver stack**: install NVIDIA's 581.42 Game Ready (or newer Studio) driver so the host exposes CUDA 13.0 capabilities required by the RTX 5090/Blackwell GPUs. On Linux download the matching 581.42 `.run` installer or vendor package from NVIDIA's [driver portal](https://www.nvidia.com/Download/index.aspx); on Windows update via GeForce Experience or the same portal. Reboot after the upgrade so the kernel picks up the new module.

4. **Verify host access**:

   ```bash
   docker run --gpus all --rm nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```

   The command should list your GPU (e.g., `NVIDIA GeForce RTX 5090`) and report `Driver Version: 581.42` (or newer) with `CUDA Version: 13.0`.

    > **TensorFlow compatibility:** The project now targets TensorFlow 2.20.0,
    > which ships in NVIDIA's TensorFlow Release 25.09 on top of CUDA 13.0. If you
    > build your own container image, keep the CUDA base at 13.0 or later so the
    > bundled Blackwell kernels stay compatible with the runtime driver stack and
    > the 581.42+ host driver.

5. **Run the trading agent with GPU access**:

   *Docker Compose*: the bundled `docker-compose.yml` now sets `runtime: nvidia`
   and reserves the host GPUs via the `deploy.resources.reservations.devices`
   stanza. After installing the toolkit you can launch the service with:

   ```bash
   docker compose up --build
   ```

   *Direct `docker run`*: if you prefer the raw `docker run` workflow keep the
   original `--gpus all` flag:

   ```bash
   docker run --gpus all -it --rm -v "$PWD:/app" ai-trading-agent:latest
   ```

   *Run directly from the NGC base image*: if you prefer to skip the local build and rely entirely on NVIDIA's maintained TensorFlow stack, mount the repository into the stock container and install the Python dependencies on first launch:

   ```bash
   docker pull nvcr.io/nvidia/tensorflow:25.09-tf2-py3
   docker run --gpus all -it --rm \
     -p 8050:8050 \
     -v "$PWD:/app" \
     -w /app \
     nvcr.io/nvidia/tensorflow:25.09-tf2-py3 \
     bash -lc "python -m pip install --user -r requirements.txt && ./launch_all.sh"
   ```

   The base image already exposes TensorFlow 2.20.0+, CUDA 13.0, and cuDNN 9.8 so no manual CUDA toolkit installation is required inside the container.

When GPU access is configured correctly the agent logs will note the detected
GPU instead of emitting CUDA initialisation warnings, and TensorFlow inference
will run noticeably faster.

If the runtime prints `No GPU detected. TensorFlow will run on CPU. If you
expected GPU support, confirm the NVIDIA drivers, container runtime, and '--gpus
all' flag (for Docker) are configured.`, the agent has automatically switched to
CPU execution. Verify the above checklist if you intended to run with GPU
acceleration; otherwise, you can safely ignore the message.

#### TensorFlow GPU 安装速览（中文）

针对需要中文快速入门的同事，这里简要摘录 TensorFlow 官方 GPU 安装要点：

1. **硬件要求**：Ubuntu 或 Windows 环境需要配备支持 CUDA® 的 NVIDIA 显卡，并在宿主机上安装匹配的驱动程序。
2. **推荐方式**：为避免手动配置 CUDA/cuDNN 产生的冲突，官方建议直接使用内置 GPU 支持的 TensorFlow Docker 映像（仅 Linux）。宿主机只需安装 NVIDIA® GPU 驱动即可。
3. **pip 安装**：TensorFlow 2.x 的 pip 软件包默认包含 GPU 支持，只需在已经启用 GPU 的容器或虚拟环境中执行：

   ```bash
   pip install tensorflow
   ```

   该命令会拉取当前稳定版 TensorFlow 及其依赖（包含 CUDA® 和 cuDNN）。
4. **旧版本兼容**：如果需要 TensorFlow 1.15 及更早版本，CPU 与 GPU 版本分离，可通过以下命令选择安装：

   ```bash
   pip install tensorflow==1.15      # CPU 版本
   pip install tensorflow-gpu==1.15  # GPU 版本
   ```

更多细节请参考 TensorFlow 官方 GPU 安装指南，以确保宿主机驱动、CUDA 版本与容器环境保持一致。

### Parallel ticker scheduling

The live trading loop automatically scales how many tickers are evaluated in
parallel based on the detected hardware. CPU-only deployments retain the
previous conservative limit derived from the number of available cores. When a
CUDA-compatible GPU (for example an RTX 5090) is present the agent now expands
the worker pool so an entire watchlist can be scored in a single inference
batch. Set `MAX_PARALLEL_TICKERS` to force a specific concurrency level or
adjust the GPU-specific ceiling with `GPU_PARALLEL_TICKER_CAP` (default `192`).
These environment variables provide fine-grained control if you need to tune
for API limits or staging environments with smaller datasets.


Interactive Brokers enforces a firm pacing limit of roughly 50 requests per
second. The agent now throttles outbound calls automatically and caps the
number of concurrent ticker evaluations touching IBKR at a safe default of 24.
If your account has a different allowance you can tune the limiter with
`IBKR_PARALLEL_TICKER_CAP`, `IBKR_REQUESTS_PER_SECOND`, and
`IBKR_REQUEST_WINDOW_SECONDS` to match the throughput you are entitled to while
avoiding error code 162/420 rejections.

## Automatic Retraining

If `data/historical_data.csv` is updated regularly, you can watch the file and
retrain all models automatically:

```bash
python auto_train.py
```

The script uses `watchdog` to monitor the file and invokes
`retrain_models.py` for each available model configuration whenever the data
changes.

## Cron Jobs

When the container boots, `launch_all.sh` now installs cron entries so the
agent stays healthy without manual intervention:

* `0 * * * * /app/ai_agent_cron.sh >> /app/logs/cron.log 2>&1` gracefully
  restarts the trading agent every hour so code or configuration changes are
  picked up without manual intervention.
* `0 7 * * * /app/update_historical_dataset_cron.sh >> /app/logs/cron.log 2>&1`
  appends freshly generated rows to `data/historical_data.csv` at 07:00 every
  morning without interrupting live trading.
* `0 0 * * 0 /app/cron_backup.sh >> /app/logs/cron.log 2>&1` keeps the existing
  weekly retraining and backup routine.

Cron output is aggregated in `logs/cron.log`, making it easy to audit recent
runs alongside the agent log in `logs/ai_agent_output.log`.

## Configuring live-trading parallelism

The live trading loop evaluates tickers concurrently using a thread pool. The
pool size is controlled by the `MAX_WORKERS` environment variable. When the
variable is not set the agent chooses a conservative default equal to half of
the available CPU cores (capped at 8 workers) so smaller hosts behave as before
while larger servers do not overwhelm broker/data rate limits automatically.

On a 16-core/32-thread workstation (e.g., Ryzen 9 9950X3D) the default resolves
to 8 workers. Increase the value only if you understand the trade-offs: more
threads can speed up indicator/model evaluation but also multiplies outbound
API requests, raising the risk of throttling or rejection from market data vendors and
Interactive Brokers. Decrease the value if you need to limit concurrency for
compliance or testing.


## Closing Positions

The live agent now trades stocks only and does not open new option contracts.
If you need to flatten any legacy option exposure without running the full
trading loop, use the command-line helper built into
`tsla_ai_master_final_ready.py`:

```bash
python tsla_ai_master_final_ready.py close-options \
  [--rights C P] \
  [--only-short] \
  [--require-profit]
```

When running inside Docker Compose, execute the helper in the application
container instead of your local Python environment:

```bash
docker compose run --rm tsla_trading_app \
  python tsla_ai_master_final_ready.py close-options \
  [--rights C P] \
  [--only-short] \
  [--require-profit]
```

* `--rights` (optional) narrows the contracts by option right. Provide one or
  both of `C` (calls) and `P` (puts). When omitted the helper examines every
  option position in the account.
* `--only-short` instructs the helper to only buy-to-close short positions.
* `--require-profit` only closes positions with positive unrealized PnL.

The script connects to Interactive Brokers, aggregates option positions across
all tickers, and reuses the same safeguards as the per-ticker closing logic to
submit appropriate orders.
