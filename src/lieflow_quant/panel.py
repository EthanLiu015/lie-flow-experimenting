"""Equity panel loading and daily cross-section cloud construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def zscore_row(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def zscore_features(feat: np.ndarray) -> np.ndarray:
    """Cross-sectionally z-score each feature column."""
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return ((feat - mean) / std).astype(np.float32)


def select_universe_tickers(
    tickers: list[str],
    size: np.ndarray,
    n_target: int,
    *,
    rank_by: str = "size",
) -> np.ndarray:
    """
    Choose ``n_target`` names from a valid cross-section.

    Default ranks by descending size proxy (log price) to avoid alphabetical bias.
    """
    n = min(len(tickers), n_target)
    if rank_by == "size":
        order = np.argsort(-size, kind="stable")[:n]
    elif rank_by == "alphabetical":
        order = np.arange(n)
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")
    return order


def assign_expanding_vix_regime(
    vix: np.ndarray,
    *,
    min_history: int = 60,
) -> np.ndarray:
    """Label VIX regime using quantiles from strictly prior observations only."""
    s = pd.Series(vix, dtype=float)
    prior = s.shift(1)
    q33 = prior.expanding(min_periods=min_history).quantile(0.3333)
    q66 = prior.expanding(min_periods=min_history).quantile(0.6667)

    regime = np.full(len(vix), "unknown", dtype=object)
    valid = s.notna()
    has_hist = prior.expanding(min_periods=min_history).count() >= min_history
    mid_mask = valid & ~has_hist
    regime[mid_mask.to_numpy()] = "mid"

    hist_mask = valid & has_hist
    vals = s[hist_mask].to_numpy()
    lo = q33[hist_mask].to_numpy()
    hi = q66[hist_mask].to_numpy()
    idx = np.where(hist_mask)[0]
    regime[idx[vals <= lo]] = "low"
    regime[idx[(vals > lo) & (vals <= hi)]] = "mid"
    regime[idx[vals > hi]] = "high"
    return regime


@dataclass
class DailyCrossSection:
    date: pd.Timestamp
    tickers: list[str]
    cloud: np.ndarray  # (n_stocks, 3) float32 — momentum, vol, size
    vix: float | None = None
    regime: str | None = None


def build_daily_cross_sections(
    close: pd.DataFrame,
    *,
    momentum_window: int = 20,
    vol_window: int = 20,
    min_stocks: int = 40,
    n_target: int = 50,
    pad_to_target: bool = False,
    universe_rank_by: str = "size",
    vix: pd.Series | None = None,
) -> list[DailyCrossSection]:
    """
    Build daily cross-section clouds aligned with tickers.

    Features per stock: 20d momentum, 20d realized vol, log price (size proxy),
    cross-sectionally z-scored each day. When ``len(valid) > n_target``, keeps the
    top ``n_target`` names by size proxy (not column order).
    """
    tickers = [c for c in close.columns if c != "^VIX"]
    px = close[tickers].dropna(how="all", axis=1)

    log_ret = np.log(px / px.shift(1))
    momentum = log_ret.rolling(momentum_window).sum()
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    log_cap_proxy = np.log(px.replace(0, np.nan))

    mom_arr = momentum.to_numpy(dtype=np.float32)
    vol_arr = realized_vol.to_numpy(dtype=np.float32)
    size_arr = log_cap_proxy.to_numpy(dtype=np.float32)
    ticker_names = momentum.columns.tolist()
    n_tickers = len(ticker_names)

    sections: list[DailyCrossSection] = []
    for i, date in enumerate(px.index):
        row_mom = mom_arr[i]
        row_vol = vol_arr[i]
        row_size = size_arr[i]

        mask = np.isfinite(row_mom) & np.isfinite(row_vol) & np.isfinite(row_size)
        if mask.sum() < min_stocks:
            continue

        mom = row_mom[mask]
        vol = row_vol[mask]
        size = row_size[mask]
        day_tickers = [ticker_names[j] for j in np.nonzero(mask)[0]]

        feat = np.stack([mom, vol, size], axis=1)
        feat = zscore_features(feat)

        if len(feat) < n_target:
            if pad_to_target:
                pad_n = n_target - len(feat)
                feat = np.vstack([feat, np.tile(feat[-1:], (pad_n, 1))])
                day_tickers = day_tickers + [day_tickers[-1]] * pad_n
            else:
                continue
        elif len(feat) > n_target:
            order = select_universe_tickers(
                day_tickers, size, n_target, rank_by=universe_rank_by
            )
            feat = feat[order]
            day_tickers = [day_tickers[i] for i in order]

        vix_val = None
        if vix is not None and date in vix.index:
            vix_val = float(vix.iloc[vix.index.get_loc(date)])

        sections.append(
            DailyCrossSection(
                date=pd.Timestamp(date),
                tickers=day_tickers,
                cloud=feat.astype(np.float32),
                vix=vix_val,
            )
        )

    return sections


def load_equity_panel(data_dir: Path | str) -> tuple[pd.DataFrame, pd.Series | None]:
    data_dir = Path(data_dir)
    close_path = data_dir / "close_prices.csv"
    vix_path = data_dir / "vix.csv"
    if not close_path.exists():
        raise FileNotFoundError(f"Missing {close_path}; run download_equity_panel.py")

    close = pd.read_csv(close_path, index_col=0, parse_dates=True)
    tickers = [c for c in close.columns if c != "^VIX"]
    close[tickers] = close[tickers].astype(np.float32)
    vix = None
    if vix_path.exists():
        vix = pd.read_csv(vix_path, index_col=0, parse_dates=True)["vix"]
    return close, vix


def load_cross_sections_from_panel(
    data_dir: Path | str,
    **kwargs,
) -> list[DailyCrossSection]:
    close, vix = load_equity_panel(data_dir)
    return build_daily_cross_sections(close, vix=vix, **kwargs)


def load_cross_sections_from_npy(
    data_dir: Path | str,
    close_path: Path | str | None = None,
    **kwargs,
) -> list[DailyCrossSection]:
    """
    Load pre-built clouds.npy and attach tickers by replaying panel logic.

    Tickers are not stored in older metadata files; we rebuild alignment from
    close prices using the same feature filters as ``build_cross_section_clouds``.
    """
    data_dir = Path(data_dir)
    clouds = np.load(data_dir / "clouds.npy")
    meta = np.load(data_dir / "metadata.npz", allow_pickle=True)
    dates = pd.to_datetime(meta["dates"])

    if "tickers" in meta.files:
        sections: list[DailyCrossSection] = []
        regimes = meta["regime"] if "regime" in meta.files else None
        vix_arr = meta["vix"] if "vix" in meta.files else None
        tickers_arr = meta["tickers"]
        for i, date in enumerate(dates):
            vix_val = (
                float(vix_arr[i])
                if vix_arr is not None and not np.isnan(vix_arr[i])
                else None
            )
            regime = str(regimes[i]) if regimes is not None else None
            sections.append(
                DailyCrossSection(
                    date=pd.Timestamp(date).normalize(),
                    tickers=list(tickers_arr[i]),
                    cloud=clouds[i].astype(np.float32),
                    vix=vix_val,
                    regime=regime,
                )
            )
        return sections

    panel_dir = Path(close_path).parent if close_path is not None else data_dir
    close, vix = load_equity_panel(panel_dir)

    rebuilt = build_daily_cross_sections(close, vix=vix, **kwargs)
    rebuilt_by_date = {s.date.normalize(): s for s in rebuilt}

    sections: list[DailyCrossSection] = []
    regimes = meta["regime"] if "regime" in meta.files else None
    vix_arr = meta["vix"] if "vix" in meta.files else None

    for i, date in enumerate(dates):
        key = pd.Timestamp(date).normalize()
        if key not in rebuilt_by_date:
            continue
        src = rebuilt_by_date[key]
        regime = str(regimes[i]) if regimes is not None else None
        vix_val = float(vix_arr[i]) if vix_arr is not None and not np.isnan(vix_arr[i]) else src.vix
        sections.append(
            DailyCrossSection(
                date=key,
                tickers=src.tickers,
                cloud=clouds[i].astype(np.float32),
                vix=vix_val,
                regime=regime,
            )
        )
    return sections


def compute_forward_returns(
    close: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """Close-to-close forward returns over ``horizon`` trading days."""
    px = close[[c for c in close.columns if c != "^VIX"]]
    return px.pct_change(horizon).shift(-horizon)
