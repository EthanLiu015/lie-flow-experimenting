"""Evaluation utilities for symmetry recovery experiments."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


C4_ANGLES_DEG = np.array([0.0, 90.0, 180.0, 270.0])


def angle_histogram_peaks(
    angles_deg: np.ndarray,
    tolerance_deg: float = 20.0,
    min_peak_height_ratio: float = 0.15,
) -> dict:
    """
    Detect peaks in a rotation-angle histogram and score C4 recovery.

    Returns dict with peak locations, C4 match score, and concentration metric.
    """
    angles = np.mod(angles_deg, 360.0)
    hist, bin_edges = np.histogram(angles, bins=72, range=(0, 360))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fold 360° wrap into 0° so peaks at 0/270° are not split across bin edges.
    hist_circ = hist.astype(float).copy()
    hist_circ[0] += hist_circ[-1]
    hist_search = hist_circ[:-1]
    centers_search = bin_centers[:-1]

    height_thresh = max(hist_search.max() * min_peak_height_ratio, 1.0)
    pad = 3
    hist_pad = np.concatenate([hist_search[-pad:], hist_search, hist_search[:pad]])
    peak_idx_pad, _ = find_peaks(hist_pad, height=height_thresh, distance=3)
    peak_idx = np.array([(int(i) - pad) % len(hist_search) for i in peak_idx_pad])
    peak_idx = np.unique(peak_idx)
    peak_angles = centers_search[peak_idx]
    peak_heights = hist_search[peak_idx]

    # Match each C4 element to nearest detected peak.
    matched = []
    for target in C4_ANGLES_DEG:
        if len(peak_angles) == 0:
            matched.append(False)
            continue
        diffs = np.abs((peak_angles - target + 180) % 360 - 180)
        matched.append(diffs.min() <= tolerance_deg)

    c4_recall = float(np.mean(matched))
    concentration = float(peak_heights.sum() / max(hist.sum(), 1))

    return {
        "peak_angles_deg": peak_angles.tolist(),
        "peak_heights": peak_heights.tolist(),
        "c4_recall": c4_recall,
        "n_peaks": int(len(peak_angles)),
        "concentration": concentration,
        "hist": hist,
        "bin_centers": bin_centers,
    }


def wasserstein_to_c4_angles(angles_deg: np.ndarray) -> float:
    """Approximate W1 between empirical angles and uniform C4 atoms."""
    import ot

    angles = np.mod(angles_deg, 360.0).reshape(-1, 1)
    targets = C4_ANGLES_DEG.reshape(-1, 1)
    # Circular distance on [0, 360).
    diff = np.abs(angles - targets.T)
    diff = np.minimum(diff, 360 - diff)
    a = np.ones(len(angles)) / len(angles)
    b = np.ones(len(targets)) / len(targets)
    return float(ot.emd2(a, b, diff))
