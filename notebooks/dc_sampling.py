"""Directional Change (DC) sampling utilities

This module converts a tick/1-min price series into event-based (intrinsic time)
Directional Change events.

Concepts
--------
- theta: DC threshold (fractional), e.g., 0.002 = 0.2%.
- DCC (Directional-Change Confirmation): occurs when price reverses >= theta from
  the last extreme.
- DCE (Directional-Change Extreme): the last extreme reached before the reversal.
- Overshoot (OS): movement after confirmation until the next confirmation.

Outputs
-------
1) events dataframe at confirmation times (intrinsic time index)

Notes
-----
- Uses close/mid prices (a single price per timestamp).
- theta can be constant or a vector per timestamp (advanced).

Typical usage
-------------
# By default includes post-DCC overshoot columns (os_after_*)
events = dc_events_from_series(price_series, theta=0.002)

# If you want only raw DCC events (no OS-after):
events_raw = dc_events_from_series(price_series, theta=0.002, add_post_overshoot=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DCEvent:
    t: pd.Timestamp
    event: str                 # 'up_dcc' or 'down_dcc'
    theta: float

    # Prices at/around the confirmation
    # t (the dataclass field) is the DCC confirmation timestamp.
    p_t: float                 # price at confirmation time (DCC)

    # Pre-confirmation extreme (DCE) that the price reversed from to trigger this DCC
    p_prior_extreme: float               # DCE price (peak before down_dcc, trough before up_dcc)
    t_prior_extreme: pd.Timestamp        # timestamp of that DCE


    # DCC move: DCE -> DCC price (this is the reversal move that triggers confirmation)
    dcc_move_log: float        # log(p_t / p_prior_extreme)
    dcc_move_ret: float        # p_t / p_prior_extreme - 1

    # Optional metadata
    n_obs_since_confirm: int   # bars since last DCC (useful as intrinsic-time duration)
    n_obs_since_extreme: int   # bars since DCE was set



def _to_series(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series with a DatetimeIndex")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")
    s = prices.astype(float).dropna().sort_index()
    if len(s) < 3:
        raise ValueError("need at least 3 non-NA price points")
    return s


def _asof_return(s: pd.Series, delta: pd.Timedelta) -> pd.Series:
    """Robust time-based return using the last observed price at or before (t - delta).

    This is more robust than pct_change(freq=...) when minute data has gaps.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("s must have a DatetimeIndex")
    s = s.sort_index()
    shifted_idx = s.index - delta
    anchor = s.reindex(shifted_idx, method="pad")
    out = pd.Series(s.values / anchor.values - 1.0, index=s.index)
    out.name = f"ret_{delta}"
    return out


def _rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI using ewm(alpha=1/period). Period is in number of observations."""
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (EMA fast - EMA slow), plus signal line and histogram."""
    close = close.astype(float)
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def dc_events_from_series(
    prices: pd.Series,
    theta: float,
    use_log: bool = True,
    add_post_overshoot: bool = True,
    tick_volume: Optional[pd.Series] = None,
    exec_lag_minutes: int = 0,
) -> pd.DataFrame:
    """Extract DC confirmation events (DCCs) from a price series.

    IMPORTANT NAMING NOTE
    ---------------------
    In classic DC literature, the *overshoot* (OS) is the continuation *after* a DCC
    in the newly confirmed direction, until the next DCC occurs.

    In this implementation, we focus on *current* DCC and post-DCC overshoot:
      - rev_move_* : extreme (DCE) -> confirmation price at the DCC (the reversal that triggers confirmation)
      - os_after_* : confirmation price -> next trend extreme (true overshoot after the DCC, computed from the next event)

    By default this function also computes true post-DCC overshoot (OS-after) using
    consecutive events (see `add_post_dcc_overshoot`). You can disable this by setting
    `add_post_overshoot=False`.
    

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by timestamp.
    theta : float
        DC threshold as a fraction (e.g., 0.002 = 0.2%). Must be > 0.
    use_log : bool
        If True, use log-ratio thresholds (recommended). If False, uses simple %.
    add_post_overshoot : bool
        If True (default), adds `os_after_log/os_after_ret` columns computed from the next event.
    tick_volume : Optional[pd.Series]
        Optional tick volume series aligned to `prices` (same DatetimeIndex). If provided, the
        output will include rolling average tick-volume features.
    exec_lag_minutes : int
        Decision/execution lag in minutes. If > 0, the output will include:
        - t_execute: t_confirm shifted forward by this lag (in calendar time)
        - p_t_execute: price at/just before t_execute (ASOF)
        - tradable_flag: True iff t_execute occurs before the next t_confirm (so the action can be executed
          without having "missed" the next event).

    Returns
    -------
    pd.DataFrame
        One row per confirmation event with DCC/DCE and move decomposition.
    """
    s = _to_series(prices)
    tv = None
    if tick_volume is not None:
        if not isinstance(tick_volume, pd.Series) or not isinstance(tick_volume.index, pd.DatetimeIndex):
            raise TypeError("tick_volume must be a pandas Series with a DatetimeIndex")
        tv = tick_volume.astype(float).reindex(s.index)
        # If there are gaps, keep NaN (caller can decide how to fill).
    if theta <= 0:
        raise ValueError("theta must be > 0")

    # Threshold in log space
    thr = np.log(1.0 + theta) if use_log else theta

    # Initialize
    t0 = s.index[0]
    p0 = float(s.iloc[0])

    # Direction state:
    # +1 means last confirmed trend is up (we are watching for down reversal)
    # -1 means last confirmed trend is down (watching for up reversal)
    #  0 means not initialized yet
    direction = 0

    t_last_confirm = t0
    p_last_confirm = p0

    t_extreme = t0
    p_extreme = p0

    n_since_confirm = 0
    n_since_extreme = 0

    events: List[DCEvent] = []

    # Helper lambdas for signed move computations
    def log_ratio(a: float, b: float) -> float:
        return float(np.log(a / b))

    for i in range(1, len(s)):
        t = s.index[i]
        p = float(s.iloc[i])

        n_since_confirm += 1
        n_since_extreme += 1

        if direction == 0:
            # We don't know trend yet. Track extremes both ways until we get a move >= theta.
            # Use p_last_confirm as anchor.
            up_move = log_ratio(p, p_last_confirm) if use_log else (p / p_last_confirm - 1.0)
            down_move = log_ratio(p_last_confirm, p) if use_log else (p_last_confirm / p - 1.0)

            if up_move >= thr:
                direction = +1
                # In an uptrend, extreme is the high.
                p_extreme, t_extreme = p, t
                n_since_extreme = 0
            elif down_move >= thr:
                direction = -1
                # In a downtrend, extreme is the low.
                p_extreme, t_extreme = p, t
                n_since_extreme = 0
            else:
                # Still no trend; keep closest extreme (optional). Here we just keep p_extreme as last seen.
                # You can comment out if you prefer the anchor to remain the first point.
                p_extreme, t_extreme = p, t
                n_since_extreme = 0
            continue

        if direction == +1:
            # Uptrend: update high extreme
            if p > p_extreme:
                p_extreme, t_extreme = p, t
                n_since_extreme = 0

            # Check reversal from high extreme (need price to drop by >= theta)
            rev = log_ratio(p_extreme, p) if use_log else (p_extreme / p - 1.0)
            if rev >= thr:
                # DOWNWARD DCC confirmed at (t, p)
                # Reversal move into the DCC: DCE(high) -> confirmation price
                rev_log = log_ratio(p, p_extreme) if use_log else (p / p_extreme - 1.0)

                events.append(
                    DCEvent(
                        t=t,
                        event="down_dcc",
                        theta=float(theta),
                        p_t=p,
                        p_prior_extreme=p_extreme,
                        t_prior_extreme=t_extreme,
                        dcc_move_log=float(rev_log if use_log else np.log(1.0 + rev_log)),
                        dcc_move_ret=float(p / p_extreme - 1.0),
                        n_obs_since_confirm=int(n_since_confirm),
                        n_obs_since_extreme=int(n_since_extreme),
                    )
                )

                # Switch to DOWNtrend
                direction = -1
                t_last_confirm, p_last_confirm = t, p
                t_extreme, p_extreme = t, p  # reset extreme to current price (new low candidate)
                n_since_confirm = 0
                n_since_extreme = 0

        elif direction == -1:
            # Downtrend: update low extreme
            if p < p_extreme:
                p_extreme, t_extreme = p, t
                n_since_extreme = 0

            # Check reversal from low extreme (need price to rise by >= theta)
            rev = log_ratio(p, p_extreme) if use_log else (p / p_extreme - 1.0)
            if rev >= thr:
                # UPWARD DCC confirmed at (t, p)
                # Reversal move into the DCC: DCE(low) -> confirmation price
                rev_log = log_ratio(p, p_extreme) if use_log else (p / p_extreme - 1.0)

                events.append(
                    DCEvent(
                        t=t,
                        event="up_dcc",
                        theta=float(theta),
                        p_t=p,
                        p_prior_extreme=p_extreme,
                        t_prior_extreme=t_extreme,
                        dcc_move_log=float(rev_log if use_log else np.log(1.0 + rev_log)),
                        dcc_move_ret=float(p / p_extreme - 1.0),
                        n_obs_since_confirm=int(n_since_confirm),
                        n_obs_since_extreme=int(n_since_extreme),
                    )
                )

                # Switch to UPtrend
                direction = +1
                t_last_confirm, p_last_confirm = t, p
                t_extreme, p_extreme = t, p  # reset extreme to current price (new high candidate)
                n_since_confirm = 0
                n_since_extreme = 0

    if not events:
        return pd.DataFrame(columns=[f.name for f in DCEvent.__dataclass_fields__.values()]).set_index(pd.Index([], name="t"))

    df_events = pd.DataFrame([e.__dict__ for e in events]).set_index("t").sort_index()

    # Explicit time column for readability (same as the index)
    df_events["t_confirm"] = df_events.index

    # -----------------------------
    # Time-aligned lookback features
    # -----------------------------
    # Returns over calendar horizons (simple returns)
    # NOTE: use ASOF-style anchors to be robust to gaps in minute data.
    r_5m  = _asof_return(s, pd.Timedelta("5min"))
    r_10m = _asof_return(s, pd.Timedelta("10min"))
    r_15m = _asof_return(s, pd.Timedelta("15min"))
    r_30m = _asof_return(s, pd.Timedelta("30min"))
    r_1h  = _asof_return(s, pd.Timedelta("1h"))
    r_1d  = _asof_return(s, pd.Timedelta("1d"))
    r_30d = _asof_return(s, pd.Timedelta(days=30))
    r_60d = _asof_return(s, pd.Timedelta(days=60))

    df_events["ret_5m"]  = r_5m.reindex(df_events.index)
    df_events["ret_10m"] = r_10m.reindex(df_events.index)
    df_events["ret_15m"] = r_15m.reindex(df_events.index)
    df_events["ret_30m"] = r_30m.reindex(df_events.index)
    df_events["ret_1h"]  = r_1h.reindex(df_events.index)
    df_events["ret_1d"]  = r_1d.reindex(df_events.index)
    df_events["ret_30d"] = r_30d.reindex(df_events.index)
    df_events["ret_60d"] = r_60d.reindex(df_events.index)

    # Volatility over the same horizons, computed from log-returns within each time window.
    logret = np.log(s).diff()
    df_events["vol_5m"]  = logret.rolling("5min",  min_periods=2).std().reindex(df_events.index)
    df_events["vol_10m"] = logret.rolling("10min", min_periods=2).std().reindex(df_events.index)
    df_events["vol_15m"] = logret.rolling("15min", min_periods=2).std().reindex(df_events.index)
    df_events["vol_30m"] = logret.rolling("30min", min_periods=2).std().reindex(df_events.index)
    df_events["vol_1h"]  = logret.rolling("1h",   min_periods=2).std().reindex(df_events.index)
    df_events["vol_1d"]  = logret.rolling("1d",  min_periods=2).std().reindex(df_events.index)
    df_events["vol_30d"] = logret.rolling("30d", min_periods=2).std().reindex(df_events.index)
    df_events["vol_60d"] = logret.rolling("60d", min_periods=2).std().reindex(df_events.index)

    # -------------------------------------------------------------
    # EMA distance dispersion (std of log-distance to EMA in window)
    # -------------------------------------------------------------
    # We compute distance = log(price / EMA_span) and then take rolling std over the same window.
    # Spans are in *minutes* (number of observations), assuming your input is 1-min bars.
    spans = {
        "5m": 5,
        "10m": 10,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "1d": 1440,
        "30d": 43200,
        "60d": 86400,
    }

    # Precompute EMA distance series
    ema_dist = {}
    for k, sp in spans.items():
        ema_k = s.ewm(span=sp, adjust=False, min_periods=sp).mean()
        ema_dist[k] = np.log(s / ema_k)

    df_events["ema_dist_std_5m"]  = ema_dist["5m"].rolling("5min",  min_periods=2).std().reindex(df_events.index)
    df_events["ema_dist_std_10m"] = ema_dist["10m"].rolling("10min", min_periods=2).std().reindex(df_events.index)
    df_events["ema_dist_std_15m"] = ema_dist["15m"].rolling("15min", min_periods=2).std().reindex(df_events.index)
    df_events["ema_dist_std_30m"] = ema_dist["30m"].rolling("30min", min_periods=2).std().reindex(df_events.index)
    df_events["ema_dist_std_1h"]  = ema_dist["1h"].rolling("1h",    min_periods=2).std().reindex(df_events.index)
    df_events["ema_dist_std_1d"]  = ema_dist["1d"].rolling("1d",    min_periods=2).std().reindex(df_events.index)
    df_events["ema_dist_std_30d"] = ema_dist["30d"].rolling("30d",  min_periods=2).std().reindex(df_events.index)
    df_events["ema_dist_std_60d"] = ema_dist["60d"].rolling("60d",  min_periods=2).std().reindex(df_events.index)

    # ------------------------------
    # Volatility compression/expansion
    # ------------------------------
    df_events["vol_ratio_5m_30m"]  = df_events["vol_5m"]  / df_events["vol_30m"].replace(0.0, np.nan)
    df_events["vol_ratio_15m_1h"]  = df_events["vol_15m"] / df_events["vol_1h"].replace(0.0, np.nan)
    df_events["vol_ratio_30m_1d"]  = df_events["vol_30m"] / df_events["vol_1d"].replace(0.0, np.nan)

    # --- RSI (Wilder) ---
    df_events["rsi_5m"]  = _rsi_wilder(s, 5).reindex(df_events.index)
    df_events["rsi_10m"] = _rsi_wilder(s, 10).reindex(df_events.index)
    df_events["rsi_15m"] = _rsi_wilder(s, 15).reindex(df_events.index)
    df_events["rsi_30m"] = _rsi_wilder(s, 30).reindex(df_events.index)
    df_events["rsi_1h"]  = _rsi_wilder(s, 60).reindex(df_events.index)
    df_events["rsi_1d"]  = _rsi_wilder(s, 1440).reindex(df_events.index)

    # --- MACD ---
    # Parameterization per horizon H minutes: fast=H/2, slow=H, signal=H/4 (rounded, minimums applied)
    def _macd_params(H: int) -> Tuple[int, int, int]:
        fast = max(3, int(round(H * 0.5)))
        slow = max(fast + 1, H)
        sig  = max(2, int(round(H * 0.25)))
        return fast, slow, sig

    for name, H in [("5m", 5), ("10m", 10), ("15m", 15), ("30m", 30), ("1h", 60), ("1d", 1440)]:
        f, sl, sg = _macd_params(H)
        m, ms, mh = _macd(s, f, sl, sg)
        df_events[f"macd_{name}"] = m.reindex(df_events.index)
        df_events[f"macd_signal_{name}"] = ms.reindex(df_events.index)
        df_events[f"macd_hist_{name}"] = mh.reindex(df_events.index)

    # Average tick volume over the same horizons (if provided)
    if tv is not None:
        df_events["tickvol_avg_5m"]  = tv.rolling("5min",  min_periods=1).mean().reindex(df_events.index)
        df_events["tickvol_avg_10m"] = tv.rolling("10min", min_periods=1).mean().reindex(df_events.index)
        df_events["tickvol_avg_15m"] = tv.rolling("15min", min_periods=1).mean().reindex(df_events.index)
        df_events["tickvol_avg_30m"] = tv.rolling("30min", min_periods=1).mean().reindex(df_events.index)
        df_events["tickvol_avg_1h"]  = tv.rolling("1h",   min_periods=1).mean().reindex(df_events.index)
        df_events["tickvol_avg_1d"]  = tv.rolling("1d",  min_periods=1).mean().reindex(df_events.index)
        df_events["tickvol_avg_30d"] = tv.rolling("30d", min_periods=1).mean().reindex(df_events.index)
        df_events["tickvol_avg_60d"] = tv.rolling("60d", min_periods=1).mean().reindex(df_events.index)

    # DCC direction label (+1 after up_dcc, -1 after down_dcc)
    df_events["dcc_dir"] = np.where(df_events["event"].eq("up_dcc"), +1, -1)

    # -----------------------------
    # Execution-time (lagged) fields
    # -----------------------------
    if exec_lag_minutes and exec_lag_minutes > 0:
        lag = pd.Timedelta(minutes=int(exec_lag_minutes))
        df_events["t_execute"] = df_events["t_confirm"] + lag

        # Price at (or last observed before) execution time
        # Reindex with pad on the original price series to be robust to missing minutes.
        # Build an index for execution times that matches the timezone of the price series.
        exec_idx = pd.to_datetime(df_events["t_execute"], utc=False)
        # Make it a DatetimeIndex so we can reliably access `.tz`
        exec_idx = pd.DatetimeIndex(exec_idx)

        if isinstance(s.index, pd.DatetimeIndex):
            s_tz = s.index.tz
            e_tz = exec_idx.tz
            if s_tz is not None and e_tz is None:
                exec_idx = exec_idx.tz_localize(s_tz)
            elif s_tz is None and e_tz is not None:
                exec_idx = exec_idx.tz_convert(None)
            elif s_tz is not None and e_tz is not None and str(s_tz) != str(e_tz):
                exec_idx = exec_idx.tz_convert(s_tz)

        # Price at (or last observed before) execution time
        p_exec = s.reindex(exec_idx, method="pad")
        df_events["p_t_execute"] = p_exec.values

        # Tradable iff execution occurs strictly before the next confirmation time
        next_confirm = df_events["t_confirm"].shift(-1)
        df_events["tradable_flag"] = df_events["t_execute"].notna() & next_confirm.notna() & (df_events["t_execute"] < next_confirm)
    else:
        df_events["t_execute"] = pd.NaT
        df_events["p_t_execute"] = np.nan
        df_events["tradable_flag"] = True

    # Convenience magnitude
    df_events["dcc_move_abs_ret"] = df_events["dcc_move_ret"].abs()

    # Post-DCC extreme (true overshoot extreme): for each row i, this is the next row's DCE.
    # This matches the common DC definition of overshoot ending at the next confirmation's extreme.
    df_events["p_extreme"] = df_events["p_prior_extreme"].shift(-1)
    df_events["t_extreme"] = df_events["t_prior_extreme"].shift(-1)

    # --- Intrinsic-time bar counts for readability ---
    # 1) Bars between DCE and its confirmation (known at confirmation time)
    df_events["n_obs_between_priorextreme_confirm"] = df_events["n_obs_since_extreme"].astype(float)

    # 2) Bars between confirmation and post-DCC extreme (overshoot duration in bars)
    #    Computed using integer positions in the original price-series index.
    pos = pd.Series(np.arange(len(s)), index=s.index)
    confirm_pos = pos.reindex(df_events.index)
    extreme_pos = pos.reindex(df_events["t_extreme"])
    df_events["n_obs_between_confirm_extreme"] = (extreme_pos.values - confirm_pos.values).astype(float)

    # Drop the old, less-specific names from the public output.
    df_events = df_events.drop(columns=["n_obs_since_confirm", "n_obs_since_extreme"])

    if add_post_overshoot:
        df_events = add_post_dcc_overshoot(df_events, use_log=use_log)

    # Reorder columns for readability: put counts + direction at the end
    preferred = [
        "event",
        "theta",
        # Lookback features (time-aligned, safe at confirmation time)
        "ret_5m",
        "ret_10m",
        "ret_15m",
        "ret_30m",
        "ret_1h",
        "ret_1d",
        "ret_30d",
        "ret_60d",
        "vol_5m",
        "vol_10m",
        "vol_15m",
        "vol_30m",
        "vol_1h",
        "vol_1d",
        "vol_30d",
        "vol_60d",
        "ema_dist_std_5m",
        "ema_dist_std_10m",
        "ema_dist_std_15m",
        "ema_dist_std_30m",
        "ema_dist_std_1h",
        "ema_dist_std_1d",
        "ema_dist_std_30d",
        "ema_dist_std_60d",
        "vol_ratio_5m_30m",
        "vol_ratio_15m_1h",
        "vol_ratio_30m_1d",
        "rsi_5m",
        "rsi_10m",
        "rsi_15m",
        "rsi_30m",
        "rsi_1h",
        "rsi_1d",
        "macd_5m",
        "macd_signal_5m",
        "macd_hist_5m",
        "macd_10m",
        "macd_signal_10m",
        "macd_hist_10m",
        "macd_15m",
        "macd_signal_15m",
        "macd_hist_15m",
        "macd_30m",
        "macd_signal_30m",
        "macd_hist_30m",
        "macd_1h",
        "macd_signal_1h",
        "macd_hist_1h",
        "macd_1d",
        "macd_signal_1d",
        "macd_hist_1d",
        "tickvol_avg_5m",
        "tickvol_avg_10m",
        "tickvol_avg_15m",
        "tickvol_avg_30m",
        "tickvol_avg_1h",
        "tickvol_avg_1d",
        "tickvol_avg_30d",
        "tickvol_avg_60d",
        # Pre-confirmation DCE
        "p_prior_extreme",
        "t_prior_extreme",
        "p_t",
        "t_confirm",
        # Execution time (optional)
        "t_execute",
        "p_t_execute",
        "tradable_flag",
        # Post-confirmation extreme anchor
        "p_extreme",
        "t_extreme",
        "dcc_move_log",
        "dcc_move_ret",
        "dcc_move_abs_ret",
        "os_after_log",
        "os_after_ret",
        "os_after_abs_ret",
        # keep metadata at the end
        "n_obs_between_priorextreme_confirm",
        "n_obs_between_confirm_extreme",
        "dcc_dir",
    ]
    # De-duplicate preferred list (can happen as the feature set grows)
    seen = set()
    preferred = [c for c in preferred if not (c in seen or seen.add(c))]

    existing = [c for c in preferred if c in df_events.columns]
    extras = [c for c in df_events.columns if c not in existing]
    df_events = df_events[existing + extras]

    # Safety: ensure unique columns (required by Parquet writers)
    if not df_events.columns.is_unique:
        df_events = df_events.loc[:, ~df_events.columns.duplicated()]

    return df_events


def add_post_dcc_overshoot(events: pd.DataFrame, use_log: bool = True) -> pd.DataFrame:
    """Add *true* post-DCC overshoot columns computed ex-post from consecutive events.

    Why this helper exists
    ----------------------
    At each DCC row, `dcc_move_*` measures the reversal **into** the confirmation:
        DCE (p_extreme) -> DCC price (p_t)

    In classic DC literature, the *overshoot* (OS) is the continuation **after** the DCC
    in the newly confirmed direction, until the next DCC occurs.

    How we compute OS-after
    -----------------------
    The extreme of the trend *after* the current DCC is stored as `p_extreme` in the
    **next** event row.

    Therefore, for each event row i:
        os_after_* = move from current p_t to next row's p_extreme

    The last row has NaN overshoot because there is no next event.
    """
    if events.empty:
        return events.copy()

    req = {"event", "p_t", "p_extreme"}
    missing = req - set(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")

    out = events.copy().sort_index()

    # p_extreme is defined as the post-DCC extreme (the next row's DCE), set in dc_events_from_series.
    next_extreme = out["p_extreme"]

    if use_log:
        out["os_after_log"] = np.log(next_extreme / out["p_t"])
        out["os_after_ret"] = next_extreme / out["p_t"] - 1.0
    else:
        out["os_after_ret"] = next_extreme / out["p_t"] - 1.0
        out["os_after_log"] = np.log(1.0 + out["os_after_ret"])

    # Convenience magnitudes
    out["os_after_abs_ret"] = out["os_after_ret"].abs()

    # NOTE: do NOT recompute/overwrite dcc_move_abs_ret here; it is created in dc_events_from_series.

    return out


# ----------------------
# Quick example helpers
# ----------------------

def example_run_on_df_1m(df_1m: pd.DataFrame, theta: float = 0.002):
    """Convenience wrapper for your minute-level OHLC dataframe."""
    if "close" not in df_1m.columns:
        raise ValueError("df_1m must contain a 'close' column")
    prices = df_1m["close"].copy()
    if not isinstance(prices.index, pd.DatetimeIndex):
        # If your timestamp is a column instead of index:
        if "timestamp" in df_1m.columns:
            prices = df_1m.set_index("timestamp")["close"].copy()
        else:
            raise ValueError("Need a DatetimeIndex or a 'timestamp' column")

    events = dc_events_from_series(
        prices,
        theta=theta,
        use_log=True,
        tick_volume=df_1m.get("tick_volume") if isinstance(df_1m, pd.DataFrame) else None,
    )
    return events


# Backwards-compatible alias (older notebooks may call this)
def example_run_on_df_1h(df_1h: pd.DataFrame, theta: float = 0.002):
    """Alias for `example_run_on_df_1m` (kept for backward compatibility)."""
    return example_run_on_df_1m(df_1h, theta=theta)
