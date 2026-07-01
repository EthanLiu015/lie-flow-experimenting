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
    vix: pd.Series | None = None,
) -> list[DailyCrossSection]:
    """
    Build daily cross-section clouds aligned with tickers.

    Features per stock: 20d momentum, 20d realized vol, log price (size proxy),
    cross-sectionally z-scored each day.
    """
    tickers = [c for c in close.columns if c != "^VIX"]
    px = close[tickers].dropna(how="all", axis=1)

    log_ret = np.log(px / px.shift(1))
    momentum = log_ret.rolling(momentum_window).sum()
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    log_cap_proxy = np.log(px.replace(0, np.nan))

    sections: list[DailyCrossSection] = []
    for date in px.index:
        row_mom = momentum.loc[date]
        row_vol = realized_vol.loc[date]
        row_size = log_cap_proxy.loc[date]

        mask = row_mom.notna() & row_vol.notna() & row_size.notna()
        if mask.sum() < min_stocks:
            continue

        day_tickers = row_mom.index[mask].tolist()
        mom = row_mom[mask].to_numpy(dtype=np.float32)
        vol = row_vol[mask].to_numpy(dtype=np.float32)
        size = row_size[mask].to_numpy(dtype=np.float32)

        feat = np.stack([mom, vol, size], axis=1)
        feat = np.array([zscore_row(feat[:, i]) for i in range(3)]).T

        if len(feat) < n_target:
            pad_n = n_target - len(feat)
            feat = np.vstack([feat, np.tile(feat[-1:], (pad_n, 1))])
            day_tickers = day_tickers + [day_tickers[-1]] * pad_n
        else:
            feat = feat[:n_target]
            day_tickers = day_tickers[:n_target]

        vix_val = None
        if vix is not None and date in vix.index:
            vix_val = float(vix.loc[date])

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
