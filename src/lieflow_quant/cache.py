"""Inference cache for fast strategy evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lieflow_quant.methodology import concentration_ratio_series

FEATURES = (
    "canonical_momentum",
    "canonical_vol",
    "radial_distance",
    "concentration",
    "z_rotation_median_deg",
)


def save_inference_cache(
    rows: list[dict],
    output_path: Path | str,
) -> None:
    """Persist long-form inference rows to compressed npz."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    arrays = {col: df[col].to_numpy() for col in df.columns if col != "ticker"}
    arrays["ticker"] = df["ticker"].astype(str).to_numpy()
    np.savez_compressed(output_path, **arrays)


def load_inference_cache(cache_path: Path | str) -> pd.DataFrame:
    """Load inference cache as a long-form DataFrame."""
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=True) as data:
        df = pd.DataFrame({k: data[k] for k in data.files})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).astype("category")
    for col in FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df


def build_signals_from_cache(
    cache: pd.DataFrame,
    *,
    signal_feature: str = "canonical_momentum",
    signal_sign: float = 1.0,
    concentration_window: int = 60,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
    signal_smoothing: int = 1,
    regime_filter: str | None = None,
    combine_concentration: bool = False,
    dedupe_tickers: bool = True,
) -> pd.DataFrame:
    """Convert cached inference to a signal panel for backtesting."""
    df = cache
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = cache.copy()
        df["date"] = pd.to_datetime(df["date"])

    if regime_filter is not None and "regime" in df.columns:
        df = df[df["regime"] == regime_filter]

    if signal_feature not in df.columns:
        raise ValueError(f"Unknown signal feature: {signal_feature}")

    df = df.copy()
    df["signal"] = df[signal_feature].astype(float) * signal_sign
    if combine_concentration:
        df["signal"] = df["signal"] * df["concentration"].astype(float)

    if dedupe_tickers:
        agg: dict[str, str] = {"signal": "mean", "concentration": "first"}
        if "regime" in df.columns:
            agg["regime"] = "first"
        if "vix" in df.columns:
            agg["vix"] = "first"
        df = df.groupby(["date", "ticker"], as_index=False).agg(agg)

    if signal_smoothing > 1:
        df = df.sort_values(["ticker", "date"])
        df["signal"] = df.groupby("ticker")["signal"].transform(
            lambda s: s.rolling(signal_smoothing, min_periods=1).mean()
        )

    # Gross exposure from daily concentration vs trailing median (prior days only).
    daily_conc = df.groupby("date", sort=True)["concentration"].first()
    ratio = concentration_ratio_series(daily_conc, window=concentration_window)
    gross = ratio.clip(min_exposure, max_exposure)
    df["gross_exposure"] = df["date"].map(gross)

    keep = ["date", "ticker", "signal", "concentration", "gross_exposure"]
    for col in ("regime", "vix"):
        if col in df.columns:
            keep.append(col)
    return df[keep].copy()
