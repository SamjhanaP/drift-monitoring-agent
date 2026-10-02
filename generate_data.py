"""
Generates synthetic sensor telemetry that mimics fab-equipment behavior:
a stable baseline, a gradual drift region, and a sudden-shift region.

Why simulate this way instead of just random noise?
Because a drift detector that only gets tested on random noise proves nothing.
We need CONTROLLED anomalies with known start/end points so we can later
verify: did our detector actually catch them, and roughly when?
"""

import numpy as np
import pandas as pd


def generate_sensor_stream(
    n_points: int = 2000,
    baseline_mean: float = 50.0,
    baseline_std: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates one sensor's readings over time.

    Timeline (as fractions of n_points):
      0.00 - 0.40 : stable baseline (normal operation)
      0.40 - 0.55 : GRADUAL drift (mean slowly increases) -- e.g. tool wear
      0.55 - 0.70 : back to stable, but at the NEW (drifted) baseline
      0.70 - 0.72 : SUDDEN shift (step change) -- e.g. recipe/tool swap
      0.72 - 1.00 : stable at the new post-shift baseline
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n_points, freq="min")

    values = np.zeros(n_points)
    mean = baseline_mean

    gradual_start = int(0.40 * n_points)
    gradual_end = int(0.55 * n_points)
    sudden_point = int(0.70 * n_points)

    for i in range(n_points):
        if gradual_start <= i < gradual_end:
            # linearly ramp the mean upward across the gradual-drift window
            progress = (i - gradual_start) / (gradual_end - gradual_start)
            mean = baseline_mean + progress * 8.0  # drifts up by 8 units total
        elif i == sudden_point:
            mean += 6.0  # instantaneous step change

        values[i] = rng.normal(loc=mean, scale=baseline_std)

    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    return df


if __name__ == "__main__":
    import os
    df = generate_sensor_stream()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sensor_stream.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} points -> {out_path}")
    print(df.describe())
