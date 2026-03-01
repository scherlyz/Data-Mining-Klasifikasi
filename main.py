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

print("Jumlah data awal:", df.shape)
print(df.head())

# ============================================
# 3. Preprocessing - Menghapus missing value
# ============================================
# Ubah karakter kosong/tidak jelas menjadi NaN
df.replace(['?', ' ', ''], np.nan, inplace=True)

df = df.dropna()

print("\nJumlah data setelah missing value dihapus:", df.shape)
print(df.head())

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

# Visualisasi Confusion Matrix
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1])
plt.yticks([0, 1])

# Menampilkan angka di dalam kotak
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.colorbar()
plt.show()

# ============================================
# 8. Evaluasi
# ============================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nAccuracy  :", round(accuracy*100,2), "%")
print("Precision :", round(precision*100,2), "%")
print("Recall    :", round(recall*100,2), "%")

# Data untuk visualisasi
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy*100, precision*100, recall*100]

# Plot bar chart
plt.figure()
plt.bar(metrics, values)
plt.title("Model Evaluation Metrics")
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)

# Menampilkan nilai di atas bar
for i in range(len(values)):
    plt.text(i, values[i], round(values[i],2), ha='center', va='bottom')

plt.show()

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