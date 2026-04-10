import os
import sys
from drift import detect_drift


def main():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    original = os.path.join(BASE_DIR, "data", "processed", "churn", "train.csv")
    drifted = os.path.join(BASE_DIR, "data", "processed", "churn_drifted_v1", "train.csv")

    _, decision = detect_drift(original, drifted)

    if decision != "No Drift":
        print("Drift detected")
        sys.exit(1)   # important for GitHub Actions
    else:
        print("No drift")
        sys.exit(0)


if __name__ == "__main__":
    main()