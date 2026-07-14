"""Warm research session: load panel + cache once, evaluate many configs fast."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import pandas as pd

from lieflow_quant.backtest import run_backtest
from lieflow_quant.methodology import aggregate_multiwindow_metrics, ml_temporal_test_start
from lieflow_quant.cache import build_signals_from_cache, load_inference_cache
from lieflow_quant.panel import compute_forward_returns, load_equity_panel
from lieflow_quant.factor_timing import build_factor_timing_signals
from lieflow_quant.multi_strategy import (
    build_combined_timing_and_overlay,
    build_multi_strategy_risk_overlay,
)
from lieflow_quant.signals_advanced import (
    build_advanced_signals,
    build_adaptive_lieflow_alpha_signals,
    build_hybrid_alpha_signals,
    build_hybrid_vol_concentration_signals,
    build_vix_concentration_regime_switcher,
)
from lieflow_quant.validation import (
    DEFAULT_PERIODS,
    build_panel_sections,
    build_raw_vol_signals_for_sections,
    evaluate_multiwindow_fast,
    evaluate_signals_in_period,
)


@dataclass(frozen=True)
class MultiWindowConfig:
    """Strategy knobs for multi-window validation."""

    strategy: str = "canonical_vol"
    signal_sign: float = 1.0
    signal_smoothing: int = 20
    concentration_window: int = 60
    min_exposure: float = 0.25
    max_exposure: float = 1.0
    min_concentration_ratio: float | None = None
    regime_filter: str | None = None
    lag: int = 1
    cost_bps: float = 10.0
    lieflow_weight: float = 0.5
    raw_vol_baseline: bool = False
    hybrid_vol: bool = False
    hybrid_alpha: bool = False
    hybrid_adaptive: bool = False
    hybrid_regime_switcher: bool = False
    factor_timing: bool = False
    multi_strategy_overlay: bool = False
    combined_timing_overlay: bool = False
    hybrid_fallback: bool = True
    timing_threshold: float = 0.5
    timing_train_end: str = "2022-12-31"
    timing_soft_gate: bool = True
    timing_min_exposure: float = 0.0
    vol_book_weight: float = 0.6
    mom_book_weight: float = 0.4
    mom_window: int = 20
    mom_signal_smoothing: int = 18
    long_n: int = 0
    short_n: int = 0
    conc_good_ratio: float = 1.0
    conc_bad_ratio: float = 0.85
    exposure_neutral: float = 0.65
    vix_boost_regimes: tuple[str, ...] | None = None
    vix_cut_regimes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SingleWindowConfig:
    """Strategy knobs for single-period eval from cache."""

    signal_feature: str = "canonical_momentum"
    signal_sign: float = 1.0
    signal_smoothing: int = 1
    concentration_window: int = 60
    min_exposure: float = 0.25
    max_exposure: float = 1.0
    regime_filter: str | None = None
    combine_concentration: bool = False
    dedupe_tickers: bool = True
    lag: int = 1
    hold_days: int = 1
    cost_bps: float = 10.0
    forward_horizon: int = 1


@dataclass
class SessionStats:
    load_ms: float
    n_cache_rows: int
    n_trading_days: int
    n_universe: int
    ml_test_start: str | None = None


@dataclass
class EvalTiming:
    build_signals_ms: float
    backtest_ms: float
    total_ms: float


_WORKER_SESSION: EvalSession | None = None


def _parallel_worker_init(data_dir: str, cache_path: str, n_target: int) -> None:
    """Load panel + cache once per worker process."""
    global _WORKER_SESSION
    _WORKER_SESSION = EvalSession(
        data_dir=data_dir,
        cache_path=cache_path,
        n_target=n_target,
    )


def _parallel_eval_job(job: tuple[int, MultiWindowConfig, bool]) -> tuple[int, dict]:
    idx, config, with_timing = job
    if _WORKER_SESSION is None:
        raise RuntimeError("parallel worker session not initialized")
    return idx, _WORKER_SESSION.evaluate_multiwindow(config, with_timing=with_timing)


def _sweep_multiwindow_parallel(
    *,
    data_dir: Path,
    cache_path: Path,
    n_target: int,
    configs: list[MultiWindowConfig],
    with_timing: bool,
    n_workers: int,
) -> list[dict]:
    jobs = [(i, cfg, with_timing) for i, cfg in enumerate(configs)]
    results: list[dict | None] = [None] * len(configs)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_parallel_worker_init,
        initargs=(str(data_dir), str(cache_path), n_target),
    ) as pool:
        for idx, metrics in pool.map(_parallel_eval_job, jobs, chunksize=1):
            results[idx] = metrics
    return results  # type: ignore[return-value]


class EvalSession:
    """
    Amortize expensive I/O and panel prep across many strategy evaluations.

    Typical quant-research pattern: load once, sweep thousands of configs in-process.
    """

    def __init__(
        self,
        *,
        data_dir: Path | str = Path("data/equity"),
        cache_path: Path | str = Path("outputs/strategy/inference_cache_full_n60.npz"),
        n_target: int = 60,
        ml_train_ratio: float = 0.8,
        inference_ml_test_only: bool = True,
    ) -> None:
        t0 = perf_counter()
        self.data_dir = Path(data_dir)
        self.cache_path = Path(cache_path)
        self.n_target = n_target
        self.ml_train_ratio = ml_train_ratio
        self.inference_ml_test_only = inference_ml_test_only

        self.close, self.vix = load_equity_panel(self.data_dir)
        self.forward_returns = compute_forward_returns(self.close)
        self.sections = build_panel_sections(
            self.close, n_target=n_target, vix=self.vix
        )
        self.section_dates = frozenset(
            pd.Timestamp(s.date).normalize() for s in self.sections
        )
        sorted_section_dates = sorted(self.section_dates)
        self.ml_test_start = ml_temporal_test_start(
            sorted_section_dates,
            train_ratio=ml_train_ratio,
        )

        cache = load_inference_cache(self.cache_path)
        cache = cache[cache["date"].isin(self.section_dates)]
        if inference_ml_test_only:
            cache = cache[cache["date"] >= self.ml_test_start]
        self.cache = cache.sort_values(["date", "ticker"], ignore_index=True)

        self.load_ms = (perf_counter() - t0) * 1000

    @property
    def stats(self) -> SessionStats:
        return SessionStats(
            load_ms=self.load_ms,
            n_cache_rows=len(self.cache),
            n_trading_days=len(self.section_dates),
            n_universe=self.n_target,
            ml_test_start=self.ml_test_start.date().isoformat(),
        )

    def build_signals(self, config: MultiWindowConfig) -> pd.DataFrame:
        if config.raw_vol_baseline:
            return build_raw_vol_signals_for_sections(
                self.close,
                self.sections,
                signal_smoothing=config.signal_smoothing,
                signal_sign=config.signal_sign,
            )
        if config.combined_timing_overlay:
            return build_combined_timing_and_overlay(
                self.cache,
                self.close,
                self.sections,
                self.forward_returns,
                vol_book_weight=config.vol_book_weight,
                mom_book_weight=config.mom_book_weight,
                vol_signal_smoothing=config.signal_smoothing,
                mom_signal_smoothing=config.mom_signal_smoothing,
                momentum_window=config.mom_window,
                concentration_window=config.concentration_window,
                train_end=config.timing_train_end,
                timing_threshold=config.timing_threshold,
                timing_min_exposure=config.timing_min_exposure,
                soft_gate=config.timing_soft_gate,
                conc_good_ratio=config.conc_good_ratio,
                conc_bad_ratio=config.conc_bad_ratio,
                min_exposure=config.min_exposure,
                max_exposure=config.max_exposure,
                lag=config.lag,
                cost_bps=config.cost_bps,
            )
        if config.factor_timing:
            raw = build_raw_vol_signals_for_sections(
                self.close,
                self.sections,
                signal_smoothing=1,
            )
            return build_factor_timing_signals(
                self.cache,
                raw,
                self.forward_returns,
                signal_smoothing=config.signal_smoothing,
                concentration_window=config.concentration_window,
                train_end=config.timing_train_end,
                timing_threshold=config.timing_threshold,
                min_exposure=config.timing_min_exposure,
                max_exposure=config.max_exposure,
                soft_gate=config.timing_soft_gate,
                fallback_to_raw_vol=config.hybrid_fallback,
                lag=config.lag,
                cost_bps=config.cost_bps,
            )
        if config.multi_strategy_overlay:
            return build_multi_strategy_risk_overlay(
                self.cache,
                self.close,
                self.sections,
                vol_book_weight=config.vol_book_weight,
                mom_book_weight=config.mom_book_weight,
                vol_signal_smoothing=config.signal_smoothing,
                mom_signal_smoothing=config.mom_signal_smoothing,
                momentum_window=config.mom_window,
                concentration_window=config.concentration_window,
                conc_good_ratio=config.conc_good_ratio,
                conc_bad_ratio=config.conc_bad_ratio,
                min_exposure=config.min_exposure,
                max_exposure=config.max_exposure,
                vix_boost_regimes=config.vix_boost_regimes,
                vix_cut_regimes=config.vix_cut_regimes,
                fallback_to_full_risk=config.hybrid_fallback,
            )
        if config.hybrid_adaptive:
            raw = build_raw_vol_signals_for_sections(
                self.close,
                self.sections,
                signal_smoothing=1,
            )
            return build_adaptive_lieflow_alpha_signals(
                self.cache,
                raw,
                lieflow_strategy=config.strategy,
                lieflow_weight=config.lieflow_weight,
                signal_sign=config.signal_sign,
                signal_smoothing=config.signal_smoothing,
                concentration_window=config.concentration_window,
                min_exposure=config.min_exposure,
                max_exposure=config.max_exposure,
                fallback_to_raw_vol=config.hybrid_fallback,
            )
        if config.hybrid_alpha:
            raw = build_raw_vol_signals_for_sections(
                self.close,
                self.sections,
                signal_smoothing=1,
            )
            return build_hybrid_alpha_signals(
                self.cache,
                raw,
                lieflow_strategy=config.strategy,
                lieflow_weight=config.lieflow_weight,
                signal_sign=config.signal_sign,
                signal_smoothing=config.signal_smoothing,
                concentration_window=config.concentration_window,
                min_exposure=config.min_exposure,
                max_exposure=config.max_exposure,
                min_concentration_ratio=config.min_concentration_ratio,
                fallback_to_raw_vol=config.hybrid_fallback,
            )
        if config.hybrid_vol:
            raw = build_raw_vol_signals_for_sections(
                self.close,
                self.sections,
                signal_smoothing=config.signal_smoothing,
            )
            return build_hybrid_vol_concentration_signals(
                self.cache,
                raw,
                lieflow_weight=config.lieflow_weight,
                signal_smoothing=config.signal_smoothing,
                concentration_window=config.concentration_window,
                min_exposure=config.min_exposure,
                max_exposure=config.max_exposure,
                min_concentration_ratio=config.min_concentration_ratio or 1.0,
                fallback_to_raw_vol=config.hybrid_fallback,
            )
        if config.hybrid_regime_switcher:
            raw = build_raw_vol_signals_for_sections(
                self.close,
                self.sections,
                signal_smoothing=1,
            )
            return build_vix_concentration_regime_switcher(
                self.cache,
                raw,
                lieflow_strategy=config.strategy,
                lieflow_weight=config.lieflow_weight,
                signal_sign=config.signal_sign,
                signal_smoothing=config.signal_smoothing,
                concentration_window=config.concentration_window,
                conc_good_ratio=config.conc_good_ratio,
                conc_bad_ratio=config.conc_bad_ratio,
                min_exposure=config.min_exposure,
                max_exposure=config.max_exposure,
                exposure_neutral=config.exposure_neutral,
                vix_boost_regimes=config.vix_boost_regimes,
                vix_cut_regimes=config.vix_cut_regimes,
                fallback_to_raw_vol=config.hybrid_fallback,
            )
        return build_advanced_signals(
            self.cache,
            strategy=config.strategy,
            signal_sign=config.signal_sign,
            signal_smoothing=config.signal_smoothing,
            concentration_window=config.concentration_window,
            min_exposure=config.min_exposure,
            max_exposure=config.max_exposure,
            min_concentration_ratio=config.min_concentration_ratio,
            regime_filter=config.regime_filter,
        )

    def evaluate_multiwindow(
        self,
        config: MultiWindowConfig,
        *,
        with_timing: bool = False,
    ) -> dict:
        t0 = perf_counter()
        signals = self.build_signals(config)
        t_build = perf_counter()

        period_metrics = evaluate_multiwindow_fast(
            signals,
            self.forward_returns,
            DEFAULT_PERIODS,
            cost_bps=config.cost_bps,
            lag=config.lag,
            long_n=config.long_n,
            short_n=config.short_n,
        )
        period_names = tuple(p.name for p in DEFAULT_PERIODS)
        agg = aggregate_multiwindow_metrics(period_metrics, period_names)
        ics = [float(pm["mean_ic"]) for pm in period_metrics.values() if pm.get("n_days", 0) > 0]

        t_end = perf_counter()
        min_sharpe = agg["min_sharpe"]
        mean_sharpe = agg["mean_sharpe"]
        mean_ic = float(sum(ics) / len(ics)) if ics else float("nan")
        all_positive = agg["n_periods_positive_sharpe"] == len(period_names)

        metrics: dict = {
            "min_sharpe": min_sharpe,
            "mean_sharpe": mean_sharpe,
            "mean_total_return": agg["mean_total_return"],
            "min_total_return": agg["min_total_return"],
            "mean_annualized_return": agg["mean_annualized_return"],
            "mean_ic": mean_ic,
            "all_windows_positive": all_positive,
            "n_periods": len(DEFAULT_PERIODS),
            "n_periods_positive": agg["n_periods_positive_sharpe"],
            "periods": period_metrics,
            "strategy": config.strategy,
            "signal_sign": config.signal_sign,
            "signal_smoothing": config.signal_smoothing,
            "cost_bps": config.cost_bps,
            "lag": config.lag,
            "n_target": self.n_target,
            "hybrid_vol": config.hybrid_vol,
            "hybrid_alpha": config.hybrid_alpha,
            "hybrid_adaptive": config.hybrid_adaptive,
            "hybrid_fallback": config.hybrid_fallback,
            "min_concentration_ratio": config.min_concentration_ratio,
            "regime_filter": config.regime_filter,
            "sharpe": min_sharpe,
        }
        if with_timing:
            metrics["_timing"] = EvalTiming(
                build_signals_ms=(t_build - t0) * 1000,
                backtest_ms=(t_end - t_build) * 1000,
                total_ms=(t_end - t0) * 1000,
            ).__dict__
        return metrics

    def evaluate_single(self, config: SingleWindowConfig) -> dict:
        signals = build_signals_from_cache(
            self.cache,
            signal_feature=config.signal_feature,
            signal_sign=config.signal_sign,
            concentration_window=config.concentration_window,
            min_exposure=config.min_exposure,
            max_exposure=config.max_exposure,
            signal_smoothing=config.signal_smoothing,
            regime_filter=config.regime_filter,
            combine_concentration=config.combine_concentration,
            dedupe_tickers=config.dedupe_tickers,
        )
        fwd = self.forward_returns
        if config.forward_horizon != 1:
            fwd = compute_forward_returns(self.close, horizon=config.forward_horizon)
        result = run_backtest(
            signals,
            fwd,
            lag=config.lag,
            cost_bps=config.cost_bps,
            hold_days=config.hold_days,
            dedupe_tickers=config.dedupe_tickers,
            forward_horizon=config.forward_horizon,
        )
        return result.metrics

    def sweep_multiwindow(
        self,
        configs: list[MultiWindowConfig],
        *,
        with_timing: bool = False,
        n_workers: int | None = None,
    ) -> list[dict]:
        """
        Evaluate many configs. Set ``n_workers > 1`` for a multi-process sweep
        (each worker loads panel+cache once, then evaluates its share).
        """
        if len(configs) <= 1:
            return [self.evaluate_multiwindow(cfg, with_timing=with_timing) for cfg in configs]

        workers = n_workers if n_workers is not None else (os.cpu_count() or 1)
        if workers <= 1:
            return [self.evaluate_multiwindow(cfg, with_timing=with_timing) for cfg in configs]

        workers = min(workers, len(configs))
        return _sweep_multiwindow_parallel(
            data_dir=self.data_dir,
            cache_path=self.cache_path,
            n_target=self.n_target,
            configs=configs,
            with_timing=with_timing,
            n_workers=workers,
        )


def config_from_namespace(args) -> MultiWindowConfig:
    """Build ``MultiWindowConfig`` from argparse / CLI namespace."""
    return MultiWindowConfig(
        strategy=getattr(args, "strategy", "canonical_vol"),
        signal_sign=getattr(args, "signal_sign", 1.0),
        signal_smoothing=getattr(args, "signal_smoothing", 20),
        concentration_window=getattr(args, "concentration_window", 60),
        min_exposure=getattr(args, "min_exposure", 0.25),
        max_exposure=getattr(args, "max_exposure", 1.0),
        min_concentration_ratio=getattr(args, "min_concentration_ratio", None),
        regime_filter=getattr(args, "regime_filter", None),
        lag=getattr(args, "lag", 1),
        cost_bps=getattr(args, "cost_bps", 10.0),
        lieflow_weight=getattr(args, "lieflow_weight", 0.5),
        raw_vol_baseline=getattr(args, "raw_vol_baseline", False),
        hybrid_vol=getattr(args, "hybrid_vol", False),
        hybrid_alpha=getattr(args, "hybrid_alpha", False),
        hybrid_adaptive=getattr(args, "hybrid_adaptive", False),
        hybrid_fallback=getattr(args, "hybrid_fallback", True),
    )


def config_from_argv(argv: list[str]) -> MultiWindowConfig:
    """Parse eval_multiwindow-style CLI flags into ``MultiWindowConfig``."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy", default="canonical_vol")
    parser.add_argument("--signal-sign", type=float, default=1.0)
    parser.add_argument("--signal-smoothing", type=int, default=20)
    parser.add_argument("--concentration-window", type=int, default=60)
    parser.add_argument("--min-exposure", type=float, default=0.25)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--min-concentration-ratio", type=float, default=None)
    parser.add_argument("--regime-filter", default=None)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--lieflow-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-vol", action="store_true")
    parser.add_argument("--raw-vol-baseline", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return config_from_namespace(args)
