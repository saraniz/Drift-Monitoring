# drift detection

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # statistical test to compare distributions (drift detection)


def get_base_dir():
    """
    Returns project root directory (safe for DVC, MLflow, pipelines)
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(path):
    return pd.read_csv(path)


# take two datasets as inputs
def preprocess(df1, df2):
    # keep only numeric columns
    # df1.select_dtypes() → selects columns based on data type
    # include=np.number → keep only numeric columns
    df1 = df1.select_dtypes(include=np.number)
    df2 = df2.select_dtypes(include=np.number)

    # align columns
    # df1.columns → list of column names in df1
    # .intersection() → finds matching columns in both datasets
    # common_cols → stores shared column names
    common_cols = df1.columns.intersection(df2.columns)

    # keep only shared columns
    df1 = df1[common_cols]
    df2 = df2[common_cols]

    return df1, df2


# df.mean() → column-wise mean
# - → difference between datasets
# np.abs() → absolute value (removes negative sign)
# Purpose - Measures how much average values changed per feature
def compute_mean_diff(df1, df2):
    return np.abs(df1.mean() - df2.mean())


# Measures change in data spread (variability)
def compute_var_diff(df1, df2):
    return np.abs(df1.var() - df2.var())


# KS test drift
def compute_ks_stat(df1, df2):
    ks_scores = []  # store ks test result for each column

    # ks_2samp() → compares distributions of two samples
    # stat → drift distance
    # p_value → significance (not used here)
    # Purpose - Measures how different distributions are
    for col in df1.columns:
        stat, p_value = ks_2samp(df1[col], df2[col])
        ks_scores.append(stat)  # Collect drift score per column

    # Average drift across all features
    return np.mean(ks_scores)


# final drift score. combine all drift signals into one score
def calculate_drift_score(df1, df2):
    mean_diff = compute_mean_diff(df1, df2)
    var_diff = compute_var_diff(df1, df2)

    # .mean() → averages all values in the Series
    mean_score = mean_diff.mean()
    var_score = var_diff.mean()

    ks_score = compute_ks_stat(df1, df2)

    # combine all
    drift_score = (mean_score + var_score + ks_score) / 3

    return drift_score, mean_score, var_score, ks_score


# Converts numeric score into readable label
def interpret_drift(score):
    if score < 0.2:
        return "No Drift"
    elif score < 0.5:
        return "Mild Drift"
    else:
        return "Strong Drift"


def detect_drift(file1, file2):
    df1 = load_data(file1)
    df2 = load_data(file2)

    df1, df2 = preprocess(df1, df2)

    drift_score, mean_score, var_score, ks_score = calculate_drift_score(df1, df2)
    decision = interpret_drift(drift_score)

    print("\n-----------------------------------")
    print(f"Comparing: {file1} vs {file2}")
    print(f"Mean Difference Score: {mean_score:.4f}")
    print(f"Variance Difference Score: {var_score:.4f}")
    print(f"KS Statistic Score: {ks_score:.4f}")
    print(f"Final Drift Score: {drift_score:.4f}")
    print(f"Drift Status: {decision}")

    return drift_score, decision


def main():

    BASE_DIR = get_base_dir()

    original = os.path.join(BASE_DIR, "data", "processed", "churn", "train.csv")

    drifted_versions = [
        os.path.join(BASE_DIR, "data", "processed", "churn_drifted_v1", "train.csv"),
        os.path.join(BASE_DIR, "data", "processed", "churn_drifted_v2", "train.csv")
    ]

    report = {}

    for drift_file in drifted_versions:
        score, status = detect_drift(original, drift_file)
        report[os.path.basename(os.path.dirname(drift_file))] = {
            "drift_score": score,
            "status": status
        }

    # save for DVC
    report_dir = os.path.join(BASE_DIR, "reports")
    report_path = os.path.join(report_dir, "drift_report.json")
    os.makedirs(report_dir, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    main()