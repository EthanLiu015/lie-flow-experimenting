"""Daily trading signal generation from LieFlow symmetry inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lieflow_quant.inference import infer_cloud_symmetry, load_so3_model
from lieflow_quant.panel import DailyCrossSection, load_cross_sections_from_npy


def generate_daily_signals(
    sections: list[DailyCrossSection],
    model,
    *,
    device: str = "cpu",
    n_steps: int = 30,
    n_mc: int = 16,
    concentration_window: int = 60,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
    signal_feature: str = "canonical_momentum",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a long-form signal panel: one row per (date, ticker).

    Columns: date, ticker, signal, concentration, gross_exposure, regime, vix
    """
    rows: list[dict] = []
    concentrations: list[float] = []

    for day_idx, section in enumerate(
        tqdm(sections, desc="Generating signals", leave=False)
    ):
        out = infer_cloud_symmetry(
            model,
            section.cloud,
            device=device,
            n_steps=n_steps,
            n_mc=n_mc,
            seed=seed + day_idx,
        )
        concentrations.append(out["concentration"])

        if len(concentrations) >= concentration_window:
            med = float(np.median(concentrations[-concentration_window:]))
        else:
            med = float(np.median(concentrations)) if concentrations else 1.0
        med = max(med, 1e-6)
        gross = float(np.clip(out["concentration"] / med, min_exposure, max_exposure))

        values = out[signal_feature]
        for ticker, val in zip(section.tickers, values):
            rows.append(
                {
                    "date": section.date,
                    "ticker": ticker,
                    "signal": float(val),
                    "concentration": out["concentration"],
                    "gross_exposure": gross,
                    "z_rotation_median_deg": out["z_rotation_median_deg"],
                    "regime": section.regime,
                    "vix": section.vix,
                }
            )

    return pd.DataFrame(rows)


def run_signal_pipeline(
    *,
    config_dir: Path,
    checkpoint: Path,
    data_dir: Path,
    output_path: Path,
    device: str = "cpu",
    test_ratio: float = 0.2,
    max_days: int | None = None,
    n_mc: int = 16,
    concentration_window: int = 60,
) -> pd.DataFrame:
    """End-to-end signal generation with optional out-of-sample tail filter."""
    data_dir = Path(data_dir)
    sections = load_cross_sections_from_npy(data_dir)

    if 0 < test_ratio < 1:
        split = int(len(sections) * (1 - test_ratio))
        sections = sections[split:]

    if max_days is not None and max_days > 0:
        sections = sections[-max_days:]

    model, n_steps, _ = load_so3_model(
        config_dir,
        checkpoint,
        device=device,
        data_path=data_dir / "clouds.npy",
        metadata_path=data_dir / "metadata.npz",
    )

    signals = generate_daily_signals(
        sections,
        model,
        device=device,
        n_steps=n_steps,
        n_mc=n_mc,
        concentration_window=concentration_window,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_path, index=False)
    return signals
