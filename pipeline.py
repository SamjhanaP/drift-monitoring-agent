"""
End-to-end pipeline. This is the file that ties the three layers together:
  1. data layer      -> generate_data.py
  2. detection layer  -> drift_detector.py (pure statistics)
  3. agent layer       -> agent_mock.py OR agent_gemini.py, controlled by
                          the USE_REAL_AGENT flag below.

Run this and you'll see the full story: raw sensor stream -> detected drift
windows -> plain-English explanations, exactly like a real monitoring
dashboard would show an engineer.
"""

import os
import pandas as pd
from drift_detector import rolling_drift_scan

# Flip this to True once you have GEMINI_API_KEY set in your environment.
# Keeping it as a flag (rather than editing the import by hand every time)
# is the same pattern real projects use for swapping providers or
# dev/prod configs -- one toggle, not a code change.
USE_REAL_AGENT = os.environ.get("GEMINI_API_KEY") is not None

if USE_REAL_AGENT:
    from agent_gemini import explain_drift_event_gemini as explain_drift_event
    print("[pipeline] GEMINI_API_KEY found -- using real Gemini agent.\n")
else:
    from agent_mock import explain_drift_event_mock as explain_drift_event
    print("[pipeline] No GEMINI_API_KEY set -- using mock agent. "
          "Set the env var to switch to real Gemini calls.\n")


def run_pipeline(csv_path: str, only_significant: bool = True):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    scan = rolling_drift_scan(df)

    if only_significant:
        # IMPORTANT: detect "new alert" transitions on the FULL scan first,
        # before filtering anything out. If we filter to significant_drift
        # rows first, any dip back to moderate/stable in between gets
        # thrown away too, and two separate real events can look like one
        # long unbroken streak -- which was a real bug we hit and fixed
        # while building this.
        is_new_alert = (scan["status"] == "significant_drift") & (
            scan["status"].shift() != "significant_drift"
        )
        events = scan[is_new_alert]
    else:
        events = scan

    print(f"Scanned {len(scan)} windows, found {len(events)} distinct alert(s).\n")

    for _, row in events.iterrows():
        event = row.to_dict()
        print("=" * 70)
        print(f"ALERT — window {event['window_start_idx']}-{event['window_end_idx']}  "
              f"(PSI={event['psi']:.2f}, {event['status']})")
        print("-" * 70)
        try:
            explanation = explain_drift_event(event)
            print(explanation)
        except Exception as e:
            # Don't let one failed API call take down the whole batch --
            # log it clearly and keep going, so you still get every alert
            # that DID succeed.
            print(f"[FAILED to generate explanation: {e}]")
        print()


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sensor_stream.csv")
    run_pipeline(data_path)
