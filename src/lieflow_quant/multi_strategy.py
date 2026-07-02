"""Multi-strategy book with LieFlow geometry risk overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lieflow_quant.factor_timing import build_lieflow_daily_features
from lieflow_quant.panel import DailyCrossSection

_VIX_REGIME_SCORE: dict[str, float] = {
    "low": 0.85,
    "mid": 1.0,
    "high": 0.55,
    "unknown": 0.75,
}


def build_momentum_signals_for_sections(
    close: pd.DataFrame,
    sections: list[DailyCrossSection],
    *,
    momentum_window: int = 20,
    signal_smoothing: int = 20,
    signal_sign: float = 1.0,
) -> pd.DataFrame:
    """Cross-sectional momentum on the same universe/dates as raw vol."""
    from lieflow_quant.validation import universe_by_date_from_sections

    dates = pd.DatetimeIndex([s.date for s in sections])
    universe = universe_by_date_from_sections(sections)
    px = close[[c for c in close.columns if c != "^VIX"]]
    mom = np.log(px / px.shift(1)).rolling(momentum_window).sum()

    dates = dates.intersection(mom.index)
    long = (
        mom.loc[dates]
        .stack(future_stack=True)
        .rename("signal")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )
    long = long.dropna(subset=["signal"])
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()

    keys = pd.MultiIndex.from_arrays(
        [
            np.repeat(list(universe.keys()), [len(v) for v in universe.values()]),
            [t for tickers in universe.values() for t in tickers],
        ],
        names=["date", "ticker"],
    )
    keys = keys.set_levels(pd.to_datetime(keys.levels[0]).normalize(), level=0)
    long = long.set_index(["date", "ticker"]).loc[
        long.set_index(["date", "ticker"]).index.intersection(keys)
    ].reset_index()

    counts = long.groupby("date")["ticker"].transform("count")
    long = long[counts >= 10].copy()
    long["signal"] = long["signal"].astype(float) * signal_sign
    long["gross_exposure"] = 1.0

    if signal_smoothing > 1:
        long = long.sort_values(["ticker", "date"])
        long["signal"] = long.groupby("ticker")["signal"].transform(
            lambda s: s.rolling(signal_smoothing, min_periods=1).mean()
        )
    return long[["date", "ticker", "signal", "gross_exposure"]].copy()


def _zscore_cross_section(s: pd.Series) -> pd.Series:
    std = float(s.std())
    if std < 1e-12:
        return s * 0.0
    return (s - s.mean()) / std


def _geometry_risk_exposure(
    features: pd.DataFrame,
    date: pd.Timestamp,
    *,
    conc_good_ratio: float,
    conc_bad_ratio: float,
    min_exposure: float,
    max_exposure: float,
    vix_boost_regimes: tuple[str, ...],
    vix_cut_regimes: tuple[str, ...],
) -> float:
    if date not in features.index:
        return 1.0
    row = features.loc[date]
    ratio = float(row.get("conc_ratio", 1.0))
    regime = str(row.get("regime", "unknown"))
    span = max(conc_good_ratio - conc_bad_ratio, 1e-6)
    conc_score = float(np.clip((ratio - conc_bad_ratio) / span, 0.0, 1.0))
    vix_score = float(_VIX_REGIME_SCORE.get(regime, 0.75))
    if regime in vix_cut_regimes:
        vix_score *= 0.65
    elif regime in vix_boost_regimes:
        vix_score = min(1.0, vix_score * 1.1)
    combined = conc_score * vix_score
    return float(min_exposure + (max_exposure - min_exposure) * combined)


def build_multi_strategy_risk_overlay(
    cache: pd.DataFrame,
    close: pd.DataFrame,
    sections: list[DailyCrossSection],
    *,
    vol_book_weight: float = 0.6,
    mom_book_weight: float = 0.4,
    vol_signal_smoothing: int = 18,
    mom_signal_smoothing: int = 18,
    momentum_window: int = 20,
    concentration_window: int = 60,
    conc_good_ratio: float = 1.0,
    conc_bad_ratio: float = 0.85,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
    vix_boost_regimes: tuple[str, ...] | None = None,
    vix_cut_regimes: tuple[str, ...] | None = None,
    fallback_to_full_risk: bool = True,
) -> pd.DataFrame:
    """
    Vol + momentum multi-book with LieFlow geometry risk thermostat.

    Alpha: z-scored blend of raw vol and momentum sleeves (fixed weights).
    Overlay: LieFlow concentration × VIX regime scales gross exposure only.
    Pre-LieFlow dates trade the blended book at full risk when fallback enabled.
    """
    boost = vix_boost_regimes or ("mid", "low")
    cut = vix_cut_regimes or ("high",)
    w_sum = max(vol_book_weight + mom_book_weight, 1e-12)
    w_vol = vol_book_weight / w_sum
    w_mom = mom_book_weight / w_sum

    from lieflow_quant.validation import build_raw_vol_signals_for_sections

    vol = build_raw_vol_signals_for_sections(
        close,
        sections,
        signal_smoothing=1,
    )
    mom = build_momentum_signals_for_sections(
        close,
        sections,
        momentum_window=momentum_window,
        signal_smoothing=1,
    )

    vol = vol.rename(columns={"signal": "vol_signal"})
    mom = mom.rename(columns={"signal": "mom_signal"})
    merged = vol.merge(mom[["date", "ticker", "mom_signal"]], on=["date", "ticker"], how="outer")
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.sort_values(["ticker", "date"])

    if vol_signal_smoothing > 1:
        merged["vol_signal"] = merged.groupby("ticker")["vol_signal"].transform(
            lambda s: s.rolling(vol_signal_smoothing, min_periods=1).mean()
        )
    if mom_signal_smoothing > 1:
        merged["mom_signal"] = merged.groupby("ticker")["mom_signal"].transform(
            lambda s: s.rolling(mom_signal_smoothing, min_periods=1).mean()
        )

    merged["vol_z"] = merged.groupby("date")["vol_signal"].transform(
        lambda s: _zscore_cross_section(s.fillna(0.0))
    )
    merged["mom_z"] = merged.groupby("date")["mom_signal"].transform(
        lambda s: _zscore_cross_section(s.fillna(0.0))
    )
    merged["signal"] = w_vol * merged["vol_z"] + w_mom * merged["mom_z"]
    merged = merged.drop(columns=["vol_z", "mom_z", "vol_signal", "mom_signal"], errors="ignore")
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()

    features = build_lieflow_daily_features(cache, concentration_window=concentration_window)
    lie_dates = frozenset(features.index)

    day_exposure: dict[pd.Timestamp, float] = {}
    for d in merged["date"].dt.normalize().unique():
        d = pd.Timestamp(d).normalize()
        if d not in lie_dates:
            day_exposure[d] = 1.0 if fallback_to_full_risk else min_exposure
            continue
        day_exposure[d] = _geometry_risk_exposure(
            features,
            d,
            conc_good_ratio=conc_good_ratio,
            conc_bad_ratio=conc_bad_ratio,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
            vix_boost_regimes=boost,
            vix_cut_regimes=cut,
        )

    merged["gross_exposure"] = merged["date"].dt.normalize().map(day_exposure).fillna(
        1.0 if fallback_to_full_risk else min_exposure
    )
    merged["concentration"] = merged["date"].dt.normalize().map(features["concentration"])
    return merged[["date", "ticker", "signal", "concentration", "gross_exposure"]].copy()


def build_combined_timing_and_overlay(
    cache: pd.DataFrame,
    close: pd.DataFrame,
    sections: list[DailyCrossSection],
    forward_returns: pd.DataFrame,
    *,
    vol_book_weight: float = 0.6,
    mom_book_weight: float = 0.4,
    vol_signal_smoothing: int = 18,
    mom_signal_smoothing: int = 18,
    momentum_window: int = 20,
    concentration_window: int = 60,
    train_end: pd.Timestamp | str = "2022-12-31",
    timing_threshold: float = 0.5,
    timing_min_exposure: float = 0.0,
    soft_gate: bool = True,
    conc_good_ratio: float = 1.0,
    conc_bad_ratio: float = 0.85,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
    fallback_to_full_risk: bool = True,
    lag: int = 1,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Fallback only: multi-strategy book with both factor timing AND geometry overlay.

    Final exposure = timing_exposure * geometry_exposure (capped at max_exposure).
    """
    from lieflow_quant.factor_timing import (
        build_factor_timing_signals,
        exposure_from_probability,
        predict_proba,
        train_factor_timing_model,
        _standardize_apply,
        FEATURE_COLS,
        factor_daily_returns,
    )
    from lieflow_quant.validation import build_raw_vol_signals_for_sections

    multi = build_multi_strategy_risk_overlay(
        cache,
        close,
        sections,
        vol_book_weight=vol_book_weight,
        mom_book_weight=mom_book_weight,
        vol_signal_smoothing=vol_signal_smoothing,
        mom_signal_smoothing=mom_signal_smoothing,
        momentum_window=momentum_window,
        concentration_window=concentration_window,
        conc_good_ratio=conc_good_ratio,
        conc_bad_ratio=conc_bad_ratio,
        min_exposure=1.0,
        max_exposure=1.0,
        fallback_to_full_risk=True,
    )

    raw_vol = build_raw_vol_signals_for_sections(close, sections, signal_smoothing=1)
    features = build_lieflow_daily_features(cache, concentration_window=concentration_window)
    factor_rets = factor_daily_returns(raw_vol, forward_returns, lag=lag, cost_bps=cost_bps)
    model = train_factor_timing_model(
        features, factor_rets, train_end=pd.Timestamp(train_end).normalize()
    )

    timing_scale: dict[pd.Timestamp, float] = {}
    lie_dates = frozenset(features.index)
    for d in multi["date"].dt.normalize().unique():
        d = pd.Timestamp(d).normalize()
        if d not in lie_dates or model is None:
            timing_scale[d] = 1.0
            continue
        feat_row = features.loc[d]
        X = np.array([[float(feat_row[c]) for c in FEATURE_COLS]], dtype=float)
        Xs = _standardize_apply(X, model["mu"], model["sigma"])
        prob = float(predict_proba(Xs, model["weights"])[0])
        timing_scale[d] = exposure_from_probability(
            prob,
            threshold=timing_threshold,
            min_exposure=timing_min_exposure,
            max_exposure=1.0,
            soft=soft_gate,
        )

    multi["gross_exposure"] = multi["date"].dt.normalize().map(
        lambda d: float(
            np.clip(
                timing_scale.get(pd.Timestamp(d).normalize(), 1.0)
                * _geometry_risk_exposure(
                    features,
                    pd.Timestamp(d).normalize(),
                    conc_good_ratio=conc_good_ratio,
                    conc_bad_ratio=conc_bad_ratio,
                    min_exposure=min_exposure,
                    max_exposure=max_exposure,
                    vix_boost_regimes=("mid", "low"),
                    vix_cut_regimes=("high",),
                ),
                min_exposure,
                max_exposure,
            )
        )
    )
    return multi[["date", "ticker", "signal", "concentration", "gross_exposure"]].copy()
