import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('kidney_disease.csv')

# Check unique values for object columns
print("Unique values in object columns:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}:")
    print(df[col].unique())

# Check for '?' or other placeholders
print("\nChecking for '?' in dataset:")
for col in df.columns:
    if df[col].dtype == 'object':
        count = df[col].apply(lambda x: x == '?' or x == '\t?').sum()
        if count > 0:
            print(f"{col}: {count}")

# Convert potential numeric columns
numeric_cols = ['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
for col in numeric_cols:
    # Force convert to numeric, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nMissing values after coercion:")
print(df[numeric_cols].isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())
