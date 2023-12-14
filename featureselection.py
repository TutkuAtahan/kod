import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import pymrmr
import matplotlib.pyplot as plt
import seaborn as sns 

# Veri setini yükleme
df = pd.read_csv('veriseti.csv', delimiter=';')  

# Eksik değerleri kontrol etme
print(df.isnull().sum())

# Kategorik değişkenleri işleme (örneğin, One-Hot Encoding)
df = pd.get_dummies(df, columns=['Marital status','Application mode','Application order','Course','Previous qualification','Previous qualification (grade)','Nacionality'])

# Eksik değerleri doldurma sadece sayısal özellikler için
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Özellikleri ölçeklendirme (örneğin, Min-Max ölçeklendirme veya Standart ölçeklendirme)
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = df.drop('Target', axis=1)
y = df['Target']

# Feature Selection - mRMR ile en iyi özellikleri seçme
selected_features = pymrmr.mRMR(X, 'MIQ', 10)  # Örnek olarak, en iyi 10 özelliği seçer
X_selected = X[selected_features]

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# SVM modelini tanımlayın
svm = SVC()

# Verileri seçilen özelliklerle model ile uyumlu hale getirin
svm.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapın
y_pred = svm.predict(X_test)

# Accuracy ve F-measure değerlerini hesaplama
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
f_measure = f1_score(y_test, y_pred, average='weighted')

print("Model Doğruluğu:", accuracy)
print("F-measure değeri:", f_measure)

class_counts = df['Target'].value_counts()


# Pasta grafiği
plt.figure(figsize=(8, 8))
class_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
plt.title('Sınıf Dağılımı - Pasta Grafiği')
plt.ylabel('')
plt.show()