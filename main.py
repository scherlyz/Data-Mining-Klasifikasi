import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv('diabetes_data_upload.csv')
print("Jumlah data awal:", df.shape)

# 2. Ubah karakter kosong/tidak jelas menjadi NaN
df.replace(['?', ' ', ''], np.nan, inplace=True)

# 3. Track/Cek kolom yang memiliki missing value
missing = df.isnull().sum()
print("\nKolom yang memiliki missing value:")
print(missing[missing > 0])

# 4. Delete (hapus) baris yang mengandung missing value
df_clean = df.dropna()

# 5. Cek jumlah data setelah dihapus
print("\nJumlah data setelah missing value dihapus:", df_clean.shape)