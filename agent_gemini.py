"""
Agent layer: takes a detected drift event (pure numbers from drift_detector.py)
and asks an LLM to explain it in plain language with a root-cause hypothesis.

This is the ONLY file that talks to an LLM. Everything upstream (data
generation, PSI detection) is plain statistics with zero API calls, keeping
the LLM boundary this narrow is a real production pattern: it's cheap, fast
to test, and easy to swap providers (Gemini <-> Claude <-> anything) without
touching the math.

NOTE (Aug 2026): Google retired the old `google-generativeai` package and the
gemini-2.0-flash model. This file uses the current `google-genai` package
(pip install google-genai) and gemini-3.6-flash, the current stable model.
"""

import os
import time
from google import genai
from google.genai import errors as genai_errors


def build_prompt(drift_event: dict) -> str:
    """
    Turns a drift event's raw numbers into a prompt. This is the whole 'agent'
    design in one function: what evidence do we hand the model, and how do we
    ask it to reason over that evidence rather than just narrate the numbers?
    """
    return f"""You are a manufacturing process monitoring assistant. A statistical
drift-detection system has flagged an anomaly in sensor telemetry. Explain this
event to a process engineer in 3-4 sentences: plain language, no jargon dump.

Evidence:
- Sensor window: index {drift_event['window_start_idx']} to {drift_event['window_end_idx']}
- Time range: {drift_event['window_start_time']} to {drift_event['window_end_time']}
- PSI (drift severity score): {drift_event['psi']:.2f}
- Classification: {drift_event['status']}
- Window mean reading: {drift_event['window_mean']:.2f} (baseline was ~50.0)

Tasks:
1. State what changed, in plain terms.
2. Give ONE most-likely hypothesis for the cause (e.g. gradual tool wear,
   sudden recipe/tool change, sensor recalibration) based on whether this
   looks like a gradual ramp or a sudden jump.
3. Recommend one concrete next action.

Keep it concise. No markdown headers, just plain prose."""


def explain_drift_event_gemini(
    drift_event: dict,
    model_name: str = "gemini-3.6-flash",
    max_retries: int = 3,
) -> str:
    """
    Real Gemini call. Requires GEMINI_API_KEY set in your environment.
    Get a free key at https://aistudio.google.com/apikey

    Includes retry with exponential backoff for TRANSIENT failures (like a
    503 "server overloaded" error) -- this is a real, common failure mode
    for any external API call, not a bug in our code. A production agent
    has to expect the API itself to occasionally be flaky and retry
    automatically rather than crash on the first hiccup.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment first.")

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(drift_event)

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text
        except genai_errors.ServerError as e:
            # 503 UNAVAILABLE / 500 etc -- the API's problem, not ours.
            # Worth retrying after a short, growing delay.
            last_error = e
            wait_seconds = 2 ** attempt  # 1s, 2s, 4s
            print(f"  [agent_gemini] Server busy (attempt {attempt + 1}/{max_retries}), "
                  f"retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    # Every retry failed -- give up cleanly with a real error rather than
    # a stack trace, so the caller can decide what to do.
    raise RuntimeError(
        f"Gemini API unavailable after {max_retries} attempts: {last_error}"
    )


if __name__ == "__main__":
    # Example drift event, taken from our real drift_scan.csv output (window 900)
    example_event = {
        "window_start_idx": 900,
        "window_end_idx": 1000,
        "window_start_time": "2026-01-01 15:00:00",
        "window_end_time": "2026-01-01 16:39:00",
        "psi": 5.04,
        "status": "significant_drift",
        "window_mean": 54.20,
    }
    print(explain_drift_event_gemini(example_event))
