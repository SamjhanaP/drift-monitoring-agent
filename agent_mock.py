"""
Mock stand-in for agent_gemini.explain_drift_event_gemini().

Same function signature and same job, turn a drift event dict into a
plain-language explanation, but instead of calling a real LLM, it uses a
few simple rules to fake a plausible-sounding response. This lets us test
and demo the FULL pipeline (data -> detection -> explanation) today, and
swap in the real Gemini call later with a one-line change.
"""


def explain_drift_event_mock(drift_event: dict) -> str:
    psi = drift_event["psi"]
    mean = drift_event["window_mean"]
    baseline = 50.0
    shift = mean - baseline

    # crude heuristic standing in for what the LLM would actually reason about
    if psi > 5:
        severity = "a severe and sustained deviation from normal operating range"
    elif psi > 1:
        severity = "a clear and growing deviation from normal operating range"
    else:
        severity = "a mild but statistically notable deviation"

    direction = "risen" if shift > 0 else "fallen"

    return (
        f"Sensor readings in this window have {direction} to an average of "
        f"{mean:.1f}, compared to the normal baseline of {baseline:.1f} — "
        f"{severity} (PSI score: {psi:.2f}). The gradual build-up pattern is "
        f"most consistent with slow tool wear or sensor recalibration drift, "
        f"rather than a single equipment failure. Recommend scheduling an "
        f"inspection of this sensor and its associated tool before the drift "
        f"progresses further.\n\n[MOCK RESPONSE - replace with real Gemini call]"
    )


if __name__ == "__main__":
    example_event = {
        "window_start_idx": 900,
        "window_end_idx": 1000,
        "window_start_time": "2026-01-01 15:00:00",
        "window_end_time": "2026-01-01 16:39:00",
        "psi": 5.04,
        "status": "significant_drift",
        "window_mean": 54.20,
    }
    print(explain_drift_event_mock(example_event))
