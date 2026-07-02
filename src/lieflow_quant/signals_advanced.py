"""Advanced LieFlow signal constructions for trading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lieflow_quant.cache import build_signals_from_cache
from lieflow_quant.methodology import concentration_ratio_series


def _cross_sectional_residual(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
) -> pd.Series:
    """Per-day OLS residual of ``y_col`` on ``x_col`` (vectorized)."""
    y = df[y_col].astype(float)
    x = df[x_col].astype(float)
    date = df["date"]
    valid = y.notna() & x.notna()
    counts = valid.groupby(date).transform("sum")
    xm = x.groupby(date).transform("mean")
    ym = y.groupby(date).transform("mean")
    xc = x - xm
    yc = y - ym
    var_x = (xc * xc).groupby(date).transform("sum")
    cov_xy = (xc * yc).groupby(date).transform("sum")
    beta = cov_xy / var_x.where(var_x >= 1e-12)
    alpha = ym - beta * xm
    resid = y - (alpha + beta * x)
    return resid.where(valid & (counts >= 5))


def _apply_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    out = df.sort_values(["ticker", "date"]).copy()
    out["signal"] = (
        out.groupby("ticker")["signal"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return out


def _apply_gross_exposure(
    df: pd.DataFrame,
    *,
    concentration_window: int = 60,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
) -> pd.DataFrame:
    daily_conc = df.groupby("date")["concentration"].first().sort_index()
    ratio = concentration_ratio_series(daily_conc, window=concentration_window)
    gross = ratio.clip(min_exposure, max_exposure)
    df = df.copy()
    df["gross_exposure"] = df["date"].map(gross)
    return df


def _concentration_gate(
    df: pd.DataFrame,
    *,
    concentration_window: int,
    min_ratio: float,
) -> pd.DataFrame:
    """Zero signal and exposure on days when concentration is below ``min_ratio`` × trailing median."""
    daily_conc = df.groupby("date")["concentration"].first().sort_index()
    ratio = concentration_ratio_series(daily_conc, window=concentration_window)
    allowed = (ratio >= min_ratio).astype(float)
    df = df.copy()
    df["gate"] = df["date"].map(allowed).fillna(0.0)
    df["signal"] = df["signal"] * df["gate"]
    return df


def build_advanced_signals(
    cache: pd.DataFrame,
    *,
    strategy: str = "canonical_vol",
    signal_sign: float = 1.0,
    signal_smoothing: int = 1,
    concentration_window: int = 60,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
    min_concentration_ratio: float | None = None,
    regime_filter: str | None = None,
    dedupe_tickers: bool = True,
) -> pd.DataFrame:
    """
    Build LieFlow trading signals using advanced constructions.

    Strategies
    ----------
    canonical_vol, canonical_momentum, radial_distance
        Direct feature (legacy path).
    mom_resid_vol
        Cross-sectional momentum orthogonal to canonical vol.
    delta_momentum, delta_vol, delta_radial
        Per-ticker day-over-day change in LieFlow feature.
    radial_gated, mom_resid_gated
        Residual/radial signals with concentration gate.
    conc_scaled_momentum
        Canonical momentum scaled by excess concentration.
    mom_minus_vol
        Simple spread: canonical_momentum - canonical_vol.
    """
    df = cache.copy()
    df["date"] = pd.to_datetime(df["date"])

    if regime_filter is not None and "regime" in df.columns:
        df = df[df["regime"] == regime_filter].copy()

    if dedupe_tickers:
        agg: dict[str, str] = {
            "canonical_momentum": "mean",
            "canonical_vol": "mean",
            "radial_distance": "mean",
            "concentration": "first",
        }
        if "regime" in df.columns:
            agg["regime"] = "first"
        if "vix" in df.columns:
            agg["vix"] = "first"
        df = df.groupby(["date", "ticker"], as_index=False).agg(
            {k: v for k, v in agg.items() if k in df.columns}
        )

    strategy = strategy.lower()

    if strategy in ("canonical_vol", "canonical_momentum", "radial_distance"):
        return build_signals_from_cache(
            cache,
            signal_feature=strategy,
            signal_sign=signal_sign,
            signal_smoothing=signal_smoothing,
            concentration_window=concentration_window,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
            regime_filter=regime_filter,
            dedupe_tickers=dedupe_tickers,
        )

    if strategy == "mom_resid_vol":
        df["signal"] = _cross_sectional_residual(df, "canonical_momentum", "canonical_vol")
    elif strategy == "mom_minus_vol":
        df["signal"] = df["canonical_momentum"] - df["canonical_vol"]
    elif strategy == "delta_momentum":
        df = df.sort_values(["ticker", "date"])
        df["signal"] = df.groupby("ticker")["canonical_momentum"].diff()
    elif strategy == "delta_vol":
        df = df.sort_values(["ticker", "date"])
        df["signal"] = df.groupby("ticker")["canonical_vol"].diff()
    elif strategy == "delta_radial":
        df = df.sort_values(["ticker", "date"])
        df["signal"] = df.groupby("ticker")["radial_distance"].diff()
    elif strategy in ("radial_gated", "radial_distance_gated"):
        df["signal"] = df["radial_distance"].astype(float)
    elif strategy == "mom_resid_gated":
        df["signal"] = _cross_sectional_residual(df, "canonical_momentum", "canonical_vol")
    elif strategy == "conc_scaled_momentum":
        daily = df.groupby("date")["concentration"].first().sort_index()
        ratio = concentration_ratio_series(daily, window=concentration_window).clip(0.0, 3.0)
        df["signal"] = df["canonical_momentum"] * df["date"].map(ratio)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df["signal"] = df["signal"].astype(float) * signal_sign
    df = df.dropna(subset=["signal"])

    if strategy in ("radial_gated", "radial_distance_gated", "mom_resid_gated") or min_concentration_ratio:
        ratio = min_concentration_ratio if min_concentration_ratio is not None else 1.0
        df = _concentration_gate(
            df,
            concentration_window=concentration_window,
            min_ratio=ratio,
        )

    df = _apply_smoothing(df, signal_smoothing)
    df = _apply_gross_exposure(
        df,
        concentration_window=concentration_window,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
    )
    if "gate" in df.columns:
        df["gross_exposure"] = df["gross_exposure"] * df["gate"]
        df = df.drop(columns=["gate"])

    keep = ["date", "ticker", "signal", "concentration", "gross_exposure"]
    for col in ("regime", "vix"):
        if col in df.columns:
            keep.append(col)
    return df[keep].copy()


def build_hybrid_vol_concentration_signals(
    cache: pd.DataFrame,
    raw_vol_signals: pd.DataFrame,
    *,
    lieflow_weight: float = 0.5,
    signal_smoothing: int = 20,
    concentration_window: int = 60,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
    min_concentration_ratio: float = 1.0,
    fallback_to_raw_vol: bool = True,
) -> pd.DataFrame:
    """
    Blend raw-vol ranks with LieFlow concentration gating.

    Uses raw vol as alpha; scales gross exposure by LieFlow symmetry strength.
    When ``fallback_to_raw_vol`` is True, dates without LieFlow inference trade
    raw vol at full exposure instead of going flat (honest pre-OOS behavior).
    """
    lie = build_advanced_signals(
        cache,
        strategy="radial_distance",
        signal_smoothing=1,
        concentration_window=concentration_window,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
    )
    lie_dates = frozenset(pd.to_datetime(lie["date"]).dt.normalize())
    lie_daily = lie.groupby("date").agg(
        concentration=("concentration", "first"),
        gross_exposure=("gross_exposure", "first"),
    )

    raw = raw_vol_signals.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.drop(columns=[c for c in ("concentration", "gross_exposure") if c in raw.columns])
    raw = raw.merge(lie_daily, left_on="date", right_index=True, how="left")

    daily_conc = lie.groupby("date")["concentration"].first().sort_index()
    if min_concentration_ratio > 0:
        ratio = concentration_ratio_series(daily_conc, window=concentration_window)
        lieflow_gate = (ratio >= min_concentration_ratio).astype(float)
    else:
        lieflow_gate = pd.Series(1.0, index=daily_conc.index)

    raw = raw.sort_values(["ticker", "date"])
    if signal_smoothing > 1:
        raw["signal"] = (
            raw.groupby("ticker")["signal"]
            .transform(lambda s: s.rolling(signal_smoothing, min_periods=1).mean())
        )

    raw["has_lieflow"] = raw["date"].dt.normalize().isin(lie_dates)
    raw["lieflow_gate"] = raw["date"].map(lieflow_gate)

    if fallback_to_raw_vol:
        raw["gate"] = np.where(raw["has_lieflow"], raw["lieflow_gate"].fillna(0.0), 1.0)
        lieflow_gross = raw["gross_exposure"].fillna(1.0) * lieflow_weight + (1.0 - lieflow_weight)
        raw["gross_exposure"] = np.where(
            raw["has_lieflow"],
            lieflow_gross * raw["gate"],
            1.0,
        )
        raw["signal"] = np.where(raw["has_lieflow"], raw["signal"] * raw["gate"], raw["signal"])
    else:
        raw["gate"] = raw["date"].map(lieflow_gate).fillna(0.0)
        base_gross = raw["gross_exposure"].fillna(1.0) * lieflow_weight + (1.0 - lieflow_weight)
        raw["gross_exposure"] = base_gross * raw["gate"]
        raw["signal"] = raw["signal"] * raw["gate"]

    return raw[["date", "ticker", "signal", "concentration", "gross_exposure"]].copy()


def build_hybrid_alpha_signals(
    cache: pd.DataFrame,
    raw_vol_signals: pd.DataFrame,
    *,
    lieflow_strategy: str = "mom_minus_vol",
    lieflow_weight: float = 0.5,
    signal_sign: float = 1.0,
    signal_smoothing: int = 20,
    concentration_window: int = 60,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
    min_concentration_ratio: float | None = None,
    fallback_to_raw_vol: bool = True,
) -> pd.DataFrame:
    """
    Blend LieFlow alpha into raw-vol cross-sectional signals.

    On LieFlow-available dates the per-ticker signal is::

        (1 - w) * raw_vol + w * lieflow_alpha

    with optional concentration-based exposure scaling. Dates without LieFlow
    inference fall back to raw vol at full exposure when ``fallback_to_raw_vol``.
    """
    if not 0.0 <= lieflow_weight <= 1.0:
        raise ValueError("lieflow_weight must be in [0, 1]")

    lie = build_advanced_signals(
        cache,
        strategy=lieflow_strategy,
        signal_sign=signal_sign,
        signal_smoothing=1,
        concentration_window=concentration_window,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
        min_concentration_ratio=None,
    )
    lie_dates = frozenset(pd.to_datetime(lie["date"]).dt.normalize())
    lie_alpha = lie[["date", "ticker", "signal", "concentration", "gross_exposure"]].rename(
        columns={
            "signal": "lieflow_signal",
            "gross_exposure": "lieflow_gross",
            "concentration": "concentration",
        }
    )

    raw = raw_vol_signals.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.drop(columns=[c for c in ("concentration", "gross_exposure") if c in raw.columns])
    raw = raw.rename(columns={"signal": "raw_signal"})
    raw = raw.merge(lie_alpha, on=["date", "ticker"], how="left")

    raw = raw.sort_values(["ticker", "date"])
    if signal_smoothing > 1:
        raw["raw_signal"] = (
            raw.groupby("ticker")["raw_signal"]
            .transform(lambda s: s.rolling(signal_smoothing, min_periods=1).mean())
        )
        raw["lieflow_signal"] = (
            raw.groupby("ticker")["lieflow_signal"]
            .transform(lambda s: s.rolling(signal_smoothing, min_periods=1).mean())
        )

    raw["has_lieflow"] = raw["date"].dt.normalize().isin(lie_dates)

    exposure_scale = np.ones(len(raw), dtype=float)
    if min_concentration_ratio is not None and min_concentration_ratio > 0:
        daily_conc = lie.groupby("date")["concentration"].first().sort_index()
        ratio = concentration_ratio_series(daily_conc, window=concentration_window)
        gate = (ratio >= min_concentration_ratio).astype(float)
        exposure_scale = raw["date"].map(gate).fillna(1.0 if fallback_to_raw_vol else 0.0).to_numpy()

    w = lieflow_weight
    blended = (1.0 - w) * raw["raw_signal"].fillna(0.0) + w * raw["lieflow_signal"].fillna(0.0)
    raw["signal"] = np.where(raw["has_lieflow"], blended, raw["raw_signal"])
    lieflow_gross = raw["lieflow_gross"].fillna(1.0) * exposure_scale
    raw["gross_exposure"] = np.where(
        raw["has_lieflow"],
        np.clip(lieflow_gross, min_exposure, max_exposure),
        1.0 if fallback_to_raw_vol else 0.0,
    )
    raw.loc[~raw["has_lieflow"] & ~fallback_to_raw_vol, "signal"] = 0.0

    return raw[["date", "ticker", "signal", "concentration", "gross_exposure"]].copy()


def build_adaptive_lieflow_alpha_signals(
    cache: pd.DataFrame,
    raw_vol_signals: pd.DataFrame,
    *,
    lieflow_strategy: str = "mom_minus_vol",
    lieflow_weight: float = 0.5,
    signal_sign: float = 1.0,
    signal_smoothing: int = 20,
    concentration_window: int = 60,
    min_exposure: float = 0.5,
    max_exposure: float = 1.0,
    fallback_to_raw_vol: bool = True,
) -> pd.DataFrame:
    """
    LieFlow alpha blend where weight scales with symmetry concentration.

    Higher concentration -> more LieFlow alpha, lower -> more raw vol.
    """
    lie = build_advanced_signals(
        cache,
        strategy=lieflow_strategy,
        signal_sign=signal_sign,
        signal_smoothing=1,
        concentration_window=concentration_window,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
        min_concentration_ratio=None,
    )
    lie_dates = frozenset(pd.to_datetime(lie["date"]).dt.normalize())
    daily_conc = lie.groupby("date")["concentration"].first().sort_index()
    conc_ratio = concentration_ratio_series(daily_conc, window=concentration_window).clip(0.0, 2.0)

    lie_alpha = lie[["date", "ticker", "signal", "concentration", "gross_exposure"]].rename(
        columns={"signal": "lieflow_signal", "gross_exposure": "lieflow_gross"}
    )

    raw = raw_vol_signals.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.drop(columns=[c for c in ("concentration", "gross_exposure") if c in raw.columns])
    raw = raw.rename(columns={"signal": "raw_signal"})
    raw = raw.merge(lie_alpha, on=["date", "ticker"], how="left")
    raw = raw.sort_values(["ticker", "date"])

    if signal_smoothing > 1:
        raw["raw_signal"] = raw.groupby("ticker")["raw_signal"].transform(
            lambda s: s.rolling(signal_smoothing, min_periods=1).mean()
        )
        raw["lieflow_signal"] = raw.groupby("ticker")["lieflow_signal"].transform(
            lambda s: s.rolling(signal_smoothing, min_periods=1).mean()
        )

    raw["has_lieflow"] = raw["date"].dt.normalize().isin(lie_dates)
    day_weight = raw["date"].map(conc_ratio).fillna(0.0)
    w = (lieflow_weight * day_weight).clip(0.0, 1.0)
    raw["blend_w"] = np.where(raw["has_lieflow"], w, 0.0)

    blended = (1.0 - raw["blend_w"]) * raw["raw_signal"].fillna(0.0) + raw["blend_w"] * raw[
        "lieflow_signal"
    ].fillna(0.0)
    raw["signal"] = np.where(raw["has_lieflow"], blended, raw["raw_signal"])
    gross = raw["lieflow_gross"].fillna(1.0)
    raw["gross_exposure"] = np.where(
        raw["has_lieflow"],
        np.clip(gross, min_exposure, max_exposure),
        1.0 if fallback_to_raw_vol else 0.0,
    )
    if not fallback_to_raw_vol:
        raw.loc[~raw["has_lieflow"], "signal"] = 0.0

    return raw[["date", "ticker", "signal", "concentration", "gross_exposure"]].copy()


_VIX_REGIME_SCORE: dict[str, float] = {
    "low": 0.85,
    "mid": 1.0,
    "high": 0.55,
    "unknown": 0.75,
}


def build_vix_concentration_regime_switcher(
    cache: pd.DataFrame,
    raw_vol_signals: pd.DataFrame,
    *,
    lieflow_strategy: str = "mom_resid_vol",
    lieflow_weight: float = 0.35,
    signal_sign: float = 1.0,
    signal_smoothing: int = 18,
    concentration_window: int = 60,
    conc_good_ratio: float = 1.0,
    conc_bad_ratio: float = 0.85,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
    exposure_neutral: float = 0.65,
    vix_boost_regimes: tuple[str, ...] | None = None,
    vix_cut_regimes: tuple[str, ...] | None = None,
    fallback_to_raw_vol: bool = True,
) -> pd.DataFrame:
    """
    VIX × LieFlow concentration regime switcher.

    - Raw vol is always the fallback alpha.
    - On LieFlow days, exposure and LieFlow alpha weight scale with
      ``concentration_ratio × VIX regime score`` (soft, not binary gate).
    - Favorable: mid/low VIX + high symmetry concentration → more LieFlow alpha.
    - Adverse: high VIX + weak concentration → cut exposure, raw vol only.
    """
    boost = vix_boost_regimes or ("mid", "low")
    cut = vix_cut_regimes or ("high",)

    lie = build_advanced_signals(
        cache,
        strategy=lieflow_strategy,
        signal_sign=signal_sign,
        signal_smoothing=1,
        concentration_window=concentration_window,
        min_exposure=min_exposure,
        max_exposure=max_exposure,
        min_concentration_ratio=None,
    )
    lie_dates = frozenset(pd.to_datetime(lie["date"]).dt.normalize())

    daily = lie.groupby("date").agg(
        concentration=("concentration", "first"),
        regime=("regime", "first") if "regime" in lie.columns else ("concentration", "first"),
    )
    if "regime" not in lie.columns:
        daily["regime"] = "unknown"
    conc_ratio = concentration_ratio_series(daily["concentration"], window=concentration_window)

    lie_alpha = lie[["date", "ticker", "signal", "concentration"]].rename(
        columns={"signal": "lieflow_signal"}
    )

    raw = raw_vol_signals.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.drop(columns=[c for c in ("concentration", "gross_exposure") if c in raw.columns])
    raw = raw.rename(columns={"signal": "raw_signal"})
    raw = raw.merge(lie_alpha, on=["date", "ticker"], how="left")
    raw = raw.sort_values(["ticker", "date"])

    if signal_smoothing > 1:
        raw["raw_signal"] = raw.groupby("ticker")["raw_signal"].transform(
            lambda s: s.rolling(signal_smoothing, min_periods=1).mean()
        )
        raw["lieflow_signal"] = raw.groupby("ticker")["lieflow_signal"].transform(
            lambda s: s.rolling(signal_smoothing, min_periods=1).mean()
        )

    raw["has_lieflow"] = raw["date"].dt.normalize().isin(lie_dates)

    span = max(conc_good_ratio - conc_bad_ratio, 1e-6)

    def _day_state(d: pd.Timestamp) -> tuple[float, float]:
        if d not in conc_ratio.index:
            return 1.0, 0.0
        ratio = float(conc_ratio.loc[d])
        regime = str(daily.loc[d, "regime"]) if d in daily.index else "unknown"
        conc_score = float(np.clip((ratio - conc_bad_ratio) / span, 0.0, 1.0))
        vix_score = float(_VIX_REGIME_SCORE.get(regime, 0.75))
        if regime in cut:
            vix_score *= 0.6
        elif regime in boost:
            vix_score = min(1.0, vix_score * 1.1)
        combined = conc_score * vix_score
        gross = float(min_exposure + (max_exposure - min_exposure) * combined)
        if combined < 0.35:
            gross = min(gross, exposure_neutral)
        alpha_w = float(lieflow_weight * combined)
        return gross, alpha_w

    day_gross: dict[pd.Timestamp, float] = {}
    day_alpha_w: dict[pd.Timestamp, float] = {}
    for d in raw["date"].dt.normalize().unique():
        g, w = _day_state(pd.Timestamp(d))
        day_gross[pd.Timestamp(d)] = g
        day_alpha_w[pd.Timestamp(d)] = w

    raw["blend_w"] = raw["date"].dt.normalize().map(day_alpha_w).fillna(0.0)
    raw["gross_exposure"] = raw["date"].dt.normalize().map(day_gross).fillna(
        1.0 if fallback_to_raw_vol else 0.0
    )

    blended = (1.0 - raw["blend_w"]) * raw["raw_signal"].fillna(0.0) + raw["blend_w"] * raw[
        "lieflow_signal"
    ].fillna(0.0)
    raw["signal"] = np.where(raw["has_lieflow"], blended, raw["raw_signal"])
    if not fallback_to_raw_vol:
        raw.loc[~raw["has_lieflow"], "signal"] = 0.0
        raw.loc[~raw["has_lieflow"], "gross_exposure"] = 0.0

    return raw[["date", "ticker", "signal", "concentration", "gross_exposure"]].copy()
