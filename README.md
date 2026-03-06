# Capstone — Bitcoin RL Algo Trading (Directional Change + PPO)

This repository contains my capstone project on **reinforcement learning (PPO)** for **Bitcoin trading**, using an **event-based Directional Change (DC)** sampling framework. The goal is to test whether DC-based features and an RL policy can produce stable decision-making vs. simple baselines under realistic high-frequency constraints (execution lag, regime shifts, etc.).

---

## Project at a glance
- Asset: BTCUSD (minute bars)
- Event clock: Directional Change (DC), θ = 0.25%
- Decision points: at DC confirmations (`t_confirm`)
- Actions: LONG / HOLD / SHORT (discrete)
- Execution: optional action lag (tested 2 minutes)
- Training: one-time split + walk-forward rolling split with warm-start + VecNormalize carry/decay

---
## 0) Background & literature inspiration

### Motivation
Bitcoin trades 24/7 and exhibits frequent regime shifts and extreme volatility. Unlike traditional assets with clearer fundamentals, crypto price dynamics are often sentiment-driven, making static rule-based strategies hard to maintain across market regimes. Reinforcement learning (RL) offers a framework to learn sequential trading decisions directly from interaction with a market environment, optimizing cumulative (risk-adjusted) performance rather than one-step price forecasts.

### Literature inspiration
Our project design is guided by several themes from recent RL trading literature:

- **Event-based sampling (Directional Change + overshoot).** Directional Change (DC) represents price action in intrinsic time: a DC event triggers when price moves by a fixed threshold and is followed by an overshoot phase until the next reversal. This filters micro-noise and segments trends into more structurally meaningful regimes.  
- **Policy-gradient / actor-critic methods for trading.** PPO and related deep RL methods have been successfully applied to trading when combined with careful reward design and validation.  
- **Sequence-aware policies (LSTM + PPO).** Prior work finds that LSTM architectures can better capture temporal patterns in BTC, and integrating LSTM into PPO improves trading performance in simulation.  
- **Robustness / overfitting control.** Deep RL agents can overfit backtests; recommended practices include time-based validation (walk-forward testing), explicit overfitting detection, and model selection/ensembling across regimes.

### References
[1] Sattarov, J., & Choi, J. (2024). *Multi-level deep Q-networks for Bitcoin trading strategies*. **Scientific Reports**. https://www.nature.com/articles/s41598-024-51408-w  
[2] Liu, et al. (2021). *Bitcoin Transaction Strategy Construction Based on Deep Reinforcement Learning*. arXiv:2109.14789. https://arxiv.org/abs/2109.14789  
[3] Gort, et al. (2022). *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting*. arXiv:2209.05559. https://arxiv.org/abs/2209.05559  
[4] Wang, & Klabjan. (2023). *An Ensemble Method of Deep Reinforcement Learning for Automated Cryptocurrency Trading*. arXiv:2309.00626. https://arxiv.org/abs/2309.00626  
[5] Rayment, G. (2025). *PhD Thesis: Spread Aware Deep Reinforcement Learning for Financial Trading (SADRL)*. https://kampouridis.net/papers/PhD_Thesis_Rayment.pdf  

## 1) Project Summary

## Research questions
1. **Does Directional Change (DC) event-based sampling improve RL trading robustness** versus fixed-interval bars for BTC at minute frequency?
2. **Can a PPO agent learn a stable long/hold/short policy** that generalizes under walk-forward (deployment-like) evaluation, including realistic execution lag?
3. **Which design choices matter most** (feature horizons, reward shaping, normalization carry/decay, and action/probability gating) for out-of-sample performance and action balance?

**Core idea:** Replace fixed-interval bars as the primary “clock” with **Directional Change events**. DC detects meaningful price moves (e.g., 0.25% threshold), then characterizes subsequent overshoot behavior and asymmetry (up vs down). The RL agent uses these event-driven features (plus short-horizon market features) to choose actions: **LONG / HOLD / SHORT**.

**What’s implemented so far**
- Directional Change event generation (confirmation vs detection concepts, overshoot, asymmetry)
- PPO agent with configurable state features (hourly baseline → enhanced with multi-horizon features)
- Trading environment with:
  - discrete action space (LONG / HOLD / SHORT)
  - **$1 initial equity**, no additional capital injections
  - configurable **decision/action lag** (tested 2 minutes)
- Evaluation experiments:
  - one-time train/test split
  - walk-forward high-frequency split (rolling windows with warm-start options)
  - robustness tests by varying train/test split and evaluation windows

---

## 2) Data & preprocessing (high level)
- Input: BTCUSD minute OHLC + tick volume proxy (+ spread when available), stored as parquet
- Cleaning: timestamp parsing, numeric coercion, sorting, duplicate checks
- Quality checks: missing-minute detection and gap summaries
- Resampling views (for diagnostics/features): 1h and 1d OHLCV aggregates

> Note: the dataset is not included in this repo. See the expected schema below.

### Expected input schema
Required columns:
- `timestamp` (datetime)
- `open`, `high`, `low`, `close` (float)
- `tick_volume` (float or int)

Optional columns:
- `spread` (float)
- `volume` (float or int)

---

## 3) Methodology

### 3.1 Directional Change (DC) sampling
- **Threshold:** 0.25% (0.0025) directional change
- Focus:
  - asymmetric behavior in **up vs down** moves
  - subsequent **overshoot** after a DC confirmation
- **Frequency selection:** compared hourly vs minute inputs to balance:
  - information richness
  - feature dimensionality / model stability

### 3.2 PPO Agent Design

#### State (observations)
Iteration history:
1) **Hourly-only inputs** (baseline)
2) Found hourly-only lacks information for high-frequency decisions
3) Expanded the state with **short-horizon + multi-horizon market features** (computed from history up to each `t_confirm`), including:
   - returns: **5m / 10m / 15m / 30m / 1h / 1d / 30d / 60d** (`ret_*`)
   - volatility: **5m / 10m / 15m / 30m / 1h / 1d / 30d / 60d** (`vol_*`)
   - volume proxy: **average tick volume** over **5m / 10m / 15m / 30m / 1h / 1d** (`tickvol_avg_*`)

In addition, the state includes **DC event descriptors** such as `event_code`, `theta`, `dcc_dir`, confirmation-move features (`dcc_move_*`), and timing features (e.g., `dt_prior_to_confirm_sec`).

Planned extensions:
- Add exogenous variables:
  - VIX, cross-asset correlation, sentiment proxies
- Upgrade policy network:
  - LSTM / Transformer / other sequence models to handle short-horizon dependencies

#### Action space
- Discrete actions: **LONG / HOLD / SHORT**
- Portfolio starts with **$1 equity**, no additional capital injections
- Planned extension:
  - partial position sizing (continuous or multi-discrete)

#### Reward function
- Tested reward variants:
  - absolute PnL vs active PnL
  - penalty/scaling approaches (to manage risk-taking / overtrading)
- Planned extension:
  - transaction costs, taxes, short borrow / financing costs

#### Regularization (behavior shaping)
- **Action gate:** regularizes action distribution to avoid collapsing into always-long or always-short
- **Probability gate:** only execute an action when confidence/probability meets criteria

#### Training approach
- One-time train/test split
- Walk-forward split (HFT-style rolling windows):
  - train/validation/test windows rolled forward by a rebalance interval
  - warm-start + normalization carry/decay options for stability

---

## 4) Evaluation Notes

What was tested:
- risk-aware evaluation vs raw returns
- action gates and reward variants
- **decision vs action lag** (tested 2-minute lag)

Key challenge:
- robustness across different market regimes is hard
- in walk-forward, eval/test windows can overlap with **high volatility** periods:
  - example: eval 1 week + test 1 week can overfit to eval
  - potential fix: **extend eval window** (e.g., 2 weeks) or shorten test window

Out-of-sample:
- Backtested a later period (post 9/15) using a one-time train/test model
- Next priority: establish a stronger baseline before claiming OOS robustness

---

## 5) Repo Structure 
- `notebooks/`
  - `BitcoinRL_btc_eda.ipynb` — data cleaning + missing-minute diagnostics (minute bars)
  - `dc_processing.ipynb` — feature preprocessing utilities for DC/RL workflow
  - `dc_sampling.py` — directional-change (DC) event extraction logic
  - `test_agent_one_time_train.ipynb` — PPO training + evaluation using one-time train/test split
  - `test_agent_wf.ipynb` — PPO walk-forward (rolling) training + evaluation
  - `requirements.txt` — pip dependencies (current environment)
  - `environment.yml` — conda environment export (Vertex AI)

- `data/`
  - `df_min_from_2025-07-16.parquet` — cleaned BTC minute dataset (sample period)
  - `DC_events_1min_from_2025-07-16.parquet` — generated DC events (sample period)

---

## 6) Quickstart (current notebook workflow)

### 6.1 Install dependencies

**If using pip:**
```bash
pip install -r notebooks/requirements.txt
```
**If using Vertex AI / conda:**
```bash
conda env create -f notebooks/environment.yml
conda activate <env_name>
```

### 6.2 Prepare data
This repo includes sample-period parquet files under `data/`:
- `data/df_min_from_2025-07-16.parquet`
- `data/DC_events_1min_from_2025-07-16.parquet`
If you are using a different period or raw dataset, place your raw minute bars locally and regenerate the processed files via the notebooks below.

### 6.3 Run notebooks in order
- `notebooks/BitcoinRL_btc_eda.ipynb` — data cleaning + missing-minute diagnostics
  - output: `data/df_min_from_2025-07-16.parquet`
- `notebooks/dc_processing.ipynb` + `notebooks/dc_sampling.py` — DC event construction + feature prep
  - output: `data/DC_events_1min_from_2025-07-16.parquet`

Choose a training mode:
- `notebooks/test_agent_one_time_train.ipynb` — PPO one-time train/test split
- `notebooks/test_agent_wf.ipynb` — PPO walk-forward (rolling) training + evaluation

---

## 7) Results 
- Performance is highly sensitive to train/test splits and volatility regimes.
- Execution lag (2 minutes) materially impacts realized performance.
- Action balance remains a challenge (policy can collapse toward dominant LONG/SHORT without gating).

See **`slides/BTC RL Trading Agent Presentation - Draft.pptx`** for detailed figures and tables.

---

## 8) Experiments & Key Configs

### 8.1 Determinism / reproducibility
- Random seed: `SEED = 2026` (Python, NumPy, Torch)
- Torch deterministic settings enabled (`cudnn.deterministic=True`, `cudnn.benchmark=False`)
- Device: CPU (`WF_DEVICE = "cpu"`)

### 8.2 Walk-forward training protocol (HFT-style)
- Source split: `WF_SOURCE_SPLIT = "test"` (train/val formed from earlier history; test executed on the selected split)
- Windowing:
  - `WF_STYLE = "rolling"`
  - Train window: `WF_TRAIN_DAYS = 90`
  - Validation window: `WF_VAL_DAYS = 10`
  - Rebalance / step forward per fold: `WF_REBALANCE_DAYS = 5`
- Minimum data guards:
  - `WF_MIN_TRAIN_ROWS = 10_000` (with auto-scaling floor logic enabled)
  - `WF_MIN_VAL_ROWS = 30`
- Per-fold training budget:
  - `WF_TOTAL_TIMESTEPS = 30_000`
- Evaluation cadence / early stop:
  - `WF_EVAL_FREQ = 3_000`
  - Stop if no improvement for `WF_PATIENCE_EVALS = 5` evals (after `WF_MIN_EVALS = 3`)
  - `WF_N_VAL_EPISODES = 5`

### 8.3 PPO hyperparameters
- Policy: `MlpPolicy`
- Network: `WF_NET_ARCH = [256, 256]`
- Learning rate: `WF_LR = 1e-4`
- Discounting: `WF_GAMMA = 0.99`
- GAE: `WF_GAE_LAMBDA = 0.95`
- Clipping: `WF_CLIP_RANGE = 0.1`
- Value loss weight: `WF_VF_COEF = 0.8`
- Target KL: `WF_TARGET_KL = 0.01`
- Parallel envs / rollout math:
  - `WF_N_ENVS = 8`
  - Total rollout per update: `WF_TOTAL_ROLLOUT = 4096`
  - Derived `WF_N_STEPS = 4096 / 8 = 512`
  - Batch size: `WF_BATCH_SIZE = 512`
- Entropy:
  - `ent_coef` set as a float at init (`ENT_COEF_START = 0.001`)
  - Entropy schedule endpoints defined (`ENT_COEF_END = 0.0001`) and applied via callback in the training loop

### 8.4 Normalization (VecNormalize)
- Observation normalization enabled: `norm_obs=True`
- Reward normalization disabled: `norm_reward=False`
- Observation clipping: `clip_obs=10.0`
- Walk-forward continuity:
  - Carry VecNormalize stats across folds: `WF_CARRY_VECNORM = True`
  - Decay carried stats: `WF_CARRY_VECNORM_DECAY = 0.95`
  - Blend carried stats toward current fold distribution: `WF_CARRY_VECNORM_DECAY_TO_FOLD = True`
  - Sync train→val normalization stats during training via a custom callback

### 8.5 Warm-starting across folds
- Warm-start enabled: `WF_WARM_START = True` (source: `WF_WARM_START_SOURCE = "selected"`)
- Conditional warm-start rules enabled:
  - require prior fold excess PnL above `WF_WARM_START_MIN_PREV_EXCESS_PNL = -0.01`
  - require gated-short fraction below `WF_WARM_START_MAX_PREV_GATED_SHORT_FRAC = 0.70`
- `WF_RESET_NUM_TIMESTEPS_ON_WARM_START = False` (keeps training timestep continuity unless cold-start)

### 8.6 Action gating / constraints 
- Gates can be modified in the notebook:
  - `USE_LONG_BIAS_GATE = False`
  - `USE_CONFIDENCE_HOLD_GATE = False`
  - `USE_SHORT_CONFIDENCE_GATE = False`

---

## 9) Baselines 
- Long-only (buy-and-hold) over the same test window
- (Planned) Execution following directional change signal

---

## 10) Assumptions & limitations
- Discrete actions only (no partial sizing yet).
- Costs/slippage are simplified unless explicitly enabled in reward/eval.
- Regime shifts (volatility spikes) can dominate short validation/test windows in walk-forward splits.

---

## 11) Roadmap (Next Steps)
Highest-priority improvements:
1. **Action balance**: reduce collapse into dominant LONG/SHORT (improve regularization + reward shaping)
2. **Transaction costs + short costs**: make results more realistic
3. **Partial positions**: move beyond discrete 3-action regime
4. **Sequence models**: LSTM/Transformer policy to better exploit minute dynamics
5. **Exogenous features**: VIX / cross-asset correlation / sentiment
6. **More robust walk-forward design**:
   - better eval/test windowing for high-vol regimes
   - more systematic split sensitivity tests

---

## 12) Reproducibility

To ensure comparable results:
- fix random seeds (numpy / torch / stable-baselines3)
- log:
  - train/eval/test date ranges
  - DC threshold and feature set version
  - execution lag
  - reward definition + penalties
  - action gating settings

---

## 13) Disclaimer

This project is for educational research purposes only and is **not financial advice**. Backtests are sensitive to assumptions (slippage, costs, data quality, regime shifts).
