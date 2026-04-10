from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR.parent / 'data' / 'churn.csv'
OUTPUT = BASE_DIR.parent / 'data' / 'churn_drifted_v2.csv'

df = pd.read_csv(DATA)

# Strong Feature Drift
df["Usage Frequency"] *= np.random.uniform(0.7, 0.9, len(df))
df["Support Calls"] += np.random.randint(3, 7, len(df))
df["Payment Delay"] += np.random.randint(5, 15, len(df))

# Strong Distribution Drift
mask = np.random.rand(len(df)) < 0.5
df.loc[mask, "Contract Length"] = "Monthly"

mask = np.random.rand(len(df)) < 0.4
df.loc[mask, "Subscription Type"] = "Basic"

#concept drift
high_risk = (df["Support Calls"] > 6) | (df["Payment Delay"] > 12)
flip_idx = df[high_risk].sample(frac=0.5, random_state=42).index
df.loc[flip_idx, "Churn"] = 1

# Some loyal users still stay
loyal = df["Tenure"] > 50
flip_idx = df[loyal].sample(frac=0.3, random_state=42).index
df.loc[flip_idx, "Churn"] = 0

# Save dataset
df.to_csv(OUTPUT, index=False)