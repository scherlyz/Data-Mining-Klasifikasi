import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv('diabetes_data_upload.csv')

print("Dimensi data sebelum preprocessing:", df.shape)

# delete missing values
df.replace(['?', ' ', ''], np.nan, inplace=True)
df_clean = df.dropna().copy()

print("Dimensi data setelah hapus missing values:", df_clean.shape)
