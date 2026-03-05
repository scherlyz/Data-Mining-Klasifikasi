import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc

# ============================================
# 2. Load Dataset
# ============================================
df = pd.read_csv('diabetes_data_upload.csv')

print("Jumlah data awal :")
print(df.head())

# ============================================
# 3. Preprocessing
#    - Menghapus missing value
# ============================================
df.replace(['?', ' ', ''], np.nan, inplace=True)

missing = df.isnull().sum()
print("\nKolom yang memiliki missing value:")
print(missing[missing > 0])

df_clean = df.dropna()

print("\nJumlah data setelah missing value dihapus:", df_clean.shape)

# ============================================
# 4. Encoding Data (karena semua kategori)
# ============================================
le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Pisahkan fitur dan target
X = df.drop('class', axis=1)
y = df['class']

# ============================================
# 5. Split Data Training & Testing (90:10)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

print("Jumlah data training :", len(X_train))
print("Jumlah data testing  :", len(X_test))

# ============================================
# 6. Model Naïve Bayes
# ============================================
model = CategoricalNB()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ============================================
# 7. Confusion Matrix
# ============================================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

# ============================================
# 8. Evaluasi
# ============================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nAccuracy  :", round(accuracy*100,2), "%")
print("Precision :", round(precision*100,2), "%")
print("Recall    :", round(recall*100,2), "%")

# ============================================
# 9. ROC Curve & AUC
# ============================================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print("AUC :", round(roc_auc,3))

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Naive Bayes")
plt.show()