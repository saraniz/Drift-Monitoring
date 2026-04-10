import os
import sys
from drift import detect_drift


def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    original = os.path.join(BASE_DIR, "data", "processed", "churn", "train.csv")
    drifted = os.path.join(BASE_DIR, "data", "processed", "churn_drifted_v1", "train.csv")

    # Run drift detection
    result, decision = detect_drift(original, drifted)

    print(f"Drift Result: {result}")
    print(f"Decision: {decision}")

    # Normalize decision (avoids string mismatch issues)
    decision = decision.strip().lower()

    if decision in ["no drift", "nodrift", "no_drift"]:
        print("No drift detected → pipeline continues")
        sys.exit(0)

    elif decision in ["minor drift", "warning"]:
        print("Minor drift detected → allowing pipeline but logging warning")
        sys.exit(0)

    else:
        print("Critical drift detected → stopping pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()