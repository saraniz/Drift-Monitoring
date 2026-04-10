# gradual behavior changes (mild drift)
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR.parent / 'data' / 'churn.csv'
OUTPUT = BASE_DIR.parent / 'data' / 'churn_drifted_v1.csv'

df = pd.read_csv(DATA)

# use uniform distribution between 1.05-1.15 and multiply. goal is to increase the usage frequency 
# Uniform distribution = all values equally likely. Mean sits exactly in the middle of the range
df['Usage Frequency'] *= np.random.uniform(1.05,1.15, len(df))
df['Total Spend'] *= np.random.uniform(1.02,1.10,len(df))

#increase the number of support calls
#len(df) mean generate value for each row...len(df) give total number of raws
df['Support Calls'] += np.random.randint(0,2,len(df))

#np.random.rand() generate random value between 0 and 1
mask = np.random.rand(len(df)) > 0.15
df.loc[mask, 'Subscription Type'] = "Standard"

risk = df['Payment Delay'] > 8
flip_idx = df[risk].sample(frac=0.2,random_state=42).index
df.loc[flip_idx, "Churn"] = 1

df.to_csv(OUTPUT, index=False)