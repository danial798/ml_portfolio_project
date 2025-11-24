import pandas as pd

try:
    df = pd.read_csv('kidney_disease.csv')
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
except Exception as e:
    print(f"Error loading dataset: {e}")
