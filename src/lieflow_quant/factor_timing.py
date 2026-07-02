"""LieFlow factor-timing: predict when a baseline factor pays off."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lieflow_quant.backtest import run_backtest
from lieflow_quant.methodology import concentration_ratio_series

_VIX_REGIME_SCORE: dict[str, float] = {
    "low": 0.85,
    "mid": 1.0,
    "high": 0.55,
    "unknown": 0.75,
}


def build_lieflow_daily_features(
    cache: pd.DataFrame,
    *,
    concentration_window: int = 60,
) -> pd.DataFrame:
    """One row per LieFlow date with geometry/regime features (no lookahead)."""
    df = cache.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    agg: dict[str, tuple[str, str]] = {
        "concentration": ("concentration", "first"),
    }
    if "regime" in df.columns:
        agg["regime"] = ("regime", "first")
    if "vix" in df.columns:
        agg["vix"] = ("vix", "first")
    if "z_rotation_median_deg" in df.columns:
        agg["z_rotation_median_deg"] = ("z_rotation_median_deg", "first")

    daily = df.groupby("date").agg(**agg)
    if "regime" not in daily.columns:
        daily["regime"] = "unknown"
    if "z_rotation_median_deg" not in daily.columns:
        daily["z_rotation_median_deg"] = 0.0

    daily["conc_ratio"] = concentration_ratio_series(
        daily["concentration"], window=concentration_window
    )
    daily["vix_score"] = daily["regime"].astype(str).map(_VIX_REGIME_SCORE).fillna(0.75)
    daily["z_rot_sin"] = np.sin(np.deg2rad(daily["z_rotation_median_deg"]))
    daily["z_rot_cos"] = np.cos(np.deg2rad(daily["z_rotation_median_deg"]))
    daily["regime_low"] = (daily["regime"].astype(str) == "low").astype(float)
    daily["regime_mid"] = (daily["regime"].astype(str) == "mid").astype(float)
    daily["regime_high"] = (daily["regime"].astype(str) == "high").astype(float)
    return daily.sort_index()


FEATURE_COLS = (
    "concentration",
    "conc_ratio",
    "vix_score",
    "z_rot_sin",
    "z_rot_cos",
    "regime_low",
    "regime_mid",
    "regime_high",
)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    epochs: int = 800,
    l2: float = 1e-3,
) -> np.ndarray:
    """Fit logistic weights (including intercept) with batch gradient descent."""
    n, d = X.shape
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)
    w = np.zeros(d + 1)
    for _ in range(epochs):
        z = Xb @ w
        z = np.clip(z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        grad = (Xb.T @ err) / n + l2 * np.r_[0.0, w[1:]]
        w -= lr * grad
    return w


def predict_proba(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([np.ones((len(X), 1)), X], axis=1)
    z = np.clip(Xb @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def factor_daily_returns(
    raw_vol_signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    lag: int = 1,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Daily net returns of the raw-vol factor book (calendar-filled)."""
    bt = run_backtest(
        raw_vol_signals,
        forward_returns,
        lag=lag,
        cost_bps=cost_bps,
        fill_calendar=True,
    )
    return bt.daily_returns


def train_factor_timing_model(
    features: pd.DataFrame,
    factor_returns: pd.Series,
    *,
    train_end: pd.Timestamp,
    lag: int = 1,
    min_train_rows: int = 120,
) -> dict | None:
    """
    Train P(vol factor return > 0 | LieFlow features) on dates <= train_end.

    Labels align factor return at trade date t with features from signal date t-lag.
    """
    bday = pd.tseries.offsets.BDay(1)
    rows: list[dict] = []
    for trade_date, ret in factor_returns.items():
        sig_date = (pd.Timestamp(trade_date) - lag * bday).normalize()
        if sig_date not in features.index:
            continue
        if sig_date > train_end:
            continue
        feat = features.loc[sig_date]
        rows.append(
            {
                "date": sig_date,
                "y": 1.0 if float(ret) > 0.0 else 0.0,
                **{c: float(feat[c]) for c in FEATURE_COLS if c in feat.index},
            }
        )
    if len(rows) < min_train_rows:
        return None

    train = pd.DataFrame(rows).sort_values("date")
    X = train[list(FEATURE_COLS)].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    Xs, mu, sigma = _standardize_fit(X)
    weights = fit_logistic(Xs, y)
    return {"weights": weights, "mu": mu, "sigma": sigma, "n_train": len(train)}


def exposure_from_probability(
    prob: float,
    *,
    threshold: float,
    min_exposure: float,
    max_exposure: float,
    soft: bool,
) -> float:
    if soft:
        # Smooth ramp around threshold in probability space.
        span = max(0.05, min(0.25, threshold * 0.4))
        score = float(np.clip((prob - (threshold - span)) / (2 * span), 0.0, 1.0))
        return float(min_exposure + (max_exposure - min_exposure) * score)
    return float(max_exposure if prob >= threshold else min_exposure)


def build_factor_timing_signals(
    cache: pd.DataFrame,
    raw_vol_signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    signal_smoothing: int = 18,
    concentration_window: int = 60,
    train_end: pd.Timestamp | str = "2022-12-31",
    timing_threshold: float = 0.5,
    min_exposure: float = 0.0,
    max_exposure: float = 1.0,
    soft_gate: bool = True,
    fallback_to_raw_vol: bool = True,
    lag: int = 1,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Raw-vol alpha with LieFlow-trained factor timing overlay.

  Training (<= train_end only): logistic model predicts whether the vol factor
  earns positive net return next day from LieFlow geometry features.
  OOS: scale gross exposure by predicted probability; pre-LieFlow dates use fallback.
    """
    train_end = pd.Timestamp(train_end).normalize()
    features = build_lieflow_daily_features(cache, concentration_window=concentration_window)
    lie_dates = frozenset(features.index)

    raw = raw_vol_signals.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw.drop(columns=[c for c in ("concentration", "gross_exposure") if c in raw.columns])
    raw = raw.sort_values(["ticker", "date"])

    if signal_smoothing > 1:
        raw["signal"] = raw.groupby("ticker")["signal"].transform(
            lambda s: s.rolling(signal_smoothing, min_periods=1).mean()
        )

    # Train on unsmoothed-ish vol book for label stability (reuse smoothed signals for trading).
    train_raw = raw_vol_signals.copy()
    train_raw["date"] = pd.to_datetime(train_raw["date"]).dt.normalize()
    factor_rets = factor_daily_returns(train_raw, forward_returns, lag=lag, cost_bps=cost_bps)
    model = train_factor_timing_model(
        features,
        factor_rets,
        train_end=train_end,
        lag=lag,
    )

    day_exposure: dict[pd.Timestamp, float] = {}
    for d in raw["date"].dt.normalize().unique():
        d = pd.Timestamp(d).normalize()
        if d not in lie_dates:
            day_exposure[d] = 1.0 if fallback_to_raw_vol else 0.0
            continue
        if model is None:
            day_exposure[d] = 1.0 if fallback_to_raw_vol else 0.0
            continue
        feat_row = features.loc[d]
        X = np.array([[float(feat_row[c]) for c in FEATURE_COLS]], dtype=float)
        Xs = _standardize_apply(X, model["mu"], model["sigma"])
        prob = float(predict_proba(Xs, model["weights"])[0])
        day_exposure[d] = exposure_from_probability(
            prob,
            threshold=timing_threshold,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
            soft=soft_gate,
        )

    raw["gross_exposure"] = raw["date"].dt.normalize().map(day_exposure).fillna(
        1.0 if fallback_to_raw_vol else 0.0
    )
    raw["concentration"] = raw["date"].dt.normalize().map(
        features["concentration"] if "concentration" in features.columns else pd.Series(dtype=float)
    )
    return raw[["date", "ticker", "signal", "concentration", "gross_exposure"]].copy()
