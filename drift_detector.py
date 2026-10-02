"""
Population Stability Index (PSI) drift detector.

Core idea: bucket a reference window's values into a histogram, bucket a
current window's values into the SAME bucket edges, then measure how much
the two histograms disagree. Small disagreement = no drift. Large
disagreement = drift.
"""

import numpy as np
import pandas as pd


def compute_psi(reference: np.ndarray, current: np.ndarray, n_buckets: int = 10) -> float:
    """
    Computes PSI between two 1D arrays of values.

    We build bucket edges from the REFERENCE data only (using quantiles, so
    each reference bucket has ~equal population). This matters: if we built
    edges from the combined data, a big shift could smear across buckets in
    a way that under-counts the divergence.
    """
    # Quantile-based bucket edges from the reference distribution.
    edges = np.quantile(reference, np.linspace(0, 1, n_buckets + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Convert to proportions. Add a tiny epsilon to avoid log(0) or divide-by-zero
    # when a bucket happens to be empty in one of the two windows.
    eps = 1e-6
    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps

    psi_per_bucket = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(np.sum(psi_per_bucket))


def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    elif psi < 0.25:
        return "moderate_drift"
    else:
        return "significant_drift"


def rolling_drift_scan(
    df: pd.DataFrame,
    reference_size: int = 200,
    window_size: int = 100,
    step_size: int = 50,
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Slides a window across the stream, computing PSI of each window against
    a FIXED reference window (the first `reference_size` points = "normal").

    Returns a DataFrame with one row per window: where it starts/ends in the
    stream, its PSI score, and its classification.
    """
    values = df[value_col].to_numpy()
    reference = values[:reference_size]

    results = []
    start = reference_size
    while start + window_size <= len(values):
        window = values[start : start + window_size]
        psi = compute_psi(reference, window)
        results.append(
            {
                "window_start_idx": start,
                "window_end_idx": start + window_size,
                "window_start_time": df["timestamp"].iloc[start],
                "window_end_time": df["timestamp"].iloc[start + window_size - 1],
                "psi": psi,
                "status": classify_psi(psi),
                "window_mean": float(window.mean()),
            }
        )
        start += step_size

    return pd.DataFrame(results)


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    df = pd.read_csv(os.path.join(data_dir, "sensor_stream.csv"), parse_dates=["timestamp"])
    scan = rolling_drift_scan(df)
    scan.to_csv(os.path.join(data_dir, "drift_scan.csv"), index=False)
    print(scan[["window_start_idx", "psi", "status", "window_mean"]].to_string(index=False))
