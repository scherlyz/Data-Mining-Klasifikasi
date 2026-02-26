import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv('diabetes_data_upload.csv')

print("Dimensi data sebelum preprocessing:", df.shape)
print("\nJumlah missing value sebelum preprocessing:")
print(df.isnull().sum())

# delete missing values
df.replace(['?', ' ', ''], np.nan, inplace=True)
df_clean = df.dropna().copy()

print("\nDimensi data setelah hapus missing values:", df_clean.shape)

print("\nJumlah missing value setelah hapus missing values:")
print(df_clean.isnull().sum())