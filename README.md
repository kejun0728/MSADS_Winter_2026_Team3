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

## 1) Project Summary

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

## 5) Repo Structure (WIP)
- `notebooks/` — EDA, DC feature engineering, PPO training experiments
- `results/` — exported plots, metrics tables, model checkpoints (if saved)
- `src/` — (planned) reusable modules for DC extraction, env, training, evaluation

---

## 6) Quickstart (WIP)

Notebook execution order (current workflow):
1. `BitcoinRL_btc_eda_jim.ipynb` — data cleaning + diagnostics
2. `test_agent_v20HFT-4-4-4.ipynb` — DC features + PPO training + walk-forward evaluation

Planned CLI workflow (to be added):
- Environment setup (`pip install -r requirements.txt`)
- Data preparation (raw data → DC events/features)
- PPO training (train/eval splits, hyperparameters)
- Evaluation/backtest (including execution lag)

---

## 7) Results snapshot (preliminary)
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

### 8.6 Action gating / constraints (current defaults)
- Gates are implemented but currently off by default in this notebook:
  - `USE_LONG_BIAS_GATE = False`
  - `USE_CONFIDENCE_HOLD_GATE = False`
  - `USE_SHORT_CONFIDENCE_GATE = False`

---

## 9) Baselines (WIP)
- Long-only (buy-and-hold) over the same test window
- (Planned) simple trend baseline (e.g., moving-average crossover)
- (Planned) random / always-hold policy sanity check

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
