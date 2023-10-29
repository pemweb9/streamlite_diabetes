# Laporan Proyek Machine Learning
### Nama : Andhika satria firmansyah
### Nim : 211351154
### Kelas : Pagi A

## Domain Proyek

Diabetes menjadi salah satu penyakit yang mematikan di dunia, termasuk di Indonesia. Diabetes dapat menyebabkan komplikasi di banyak bagian tubuh dan secara keseluruhan dapat meningkatkan risiko kematian. Salah satu cara untuk mendeteksi penyakit diabetes adalah dengan memanfaatkan algoritma machine learning. Logistic regression merupakan model klasifikasi dalam machine learning yang banyak digunakan dalam analisis klinis 

## Business Understanding

### Problem Statements

- Ketidakmungkinan bagi seseorang untuk memprediksi diabetes. 

### Goals

- mencari solusi untuk menganalisis penyakit diabetes yang diderita.

### Solution statements
- Pengembangan Platform Diabetes Prediction with Logistic Regression Berbasis Web, Solusi pertama adalah mengembangkan platform pengechekan diabetes yang mengintegrasikan data dari Kaggle.com untuk memberikan pengguna akses mudah untuk menganalisis diabetes yang diderita. 
- Model yang dihasilkan dari datasets itu menggunakan metode Linear Regression.

## Data Understanding
Dataset yang saya gunakan berasal dari Kaggle.<br> 

[Diabetes Prediction with Logistic Regression](https://www.kaggle.com/code/tanyildizderya/diabetes-prediction-with-logistic-regression/notebook).

### Variabel-variabel pada Apartment Prices for Azerbaijan Market adalah sebagai berikut:

-Pregnancies : mempresentasikan berapa kali wanita tersebut hamil selama hidupnya.

-Glucose : mempresentasikan konsentrasi glukosa plasma pada 2 jam dalam tes toleransi glukosa.

-BloodPressure : Tekanan darah adalah cara yang sangat terkenal untuk mengukur kesehatan jantung seseorang, ada juga ukuran tekanan darah yaitu diastolik dan sistolik. Dalam data ini, kita memiliki tekanan darah diastolik dalam (mm / Hg) ketika jantung rileks setelah kontraksi.

-SkinThickness : nilai yang digunakan untuk memperkirakan lemak tubuh (mm) yang diukur pada lengan kanan setengah antara proses olecranon dari siku dan proses akromial skapula.

-Insulin : tingkat insulin 2 jam insulin serum dalam satuan mu U/ml.

-BMI : Indeks Massa Tubuh berat dalam kg / tinggi dalam meter kuadrat, dan merupakan indikator kesehatan seseorang.

-DiabetesPedigreeFunction : Indikator riwayat diabetes dalam keluarga

-Age : umur.

## Data Preparation
### Data Collection
Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama dataset Diabetes Prediction with Logistic Regression, jika anda tertarik dengan datasetnya, anda bisa click link diatas.

### Data Discovery And Profiling
Untuk bagian ini, kita akan menggunakan teknik EDA. <br>
Pertama kita mengimport semua library yang dibutuhkan,
```bash
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

lanjut dengan memasukkan file csv yang telah diextract pada sebuah variable, dan melihat data paling atas dari datasetsnya
```bash
df = pd.read_csv('diabetes2.csv')
```

```bash
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

```bash
df.head()
```

```bash
df.info()
```

```bash
df.describe()
```

```bash
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```

```bash
sns.countplot(x='Outcome',data=df)
```

```bash
sns.distplot(df['Age'].dropna(),kde=True)
```

```bash
df.corr()
```

```bash
sns.heatmap(df.corr())
```

```bash
sns.pairplot(df)
```

```bash
plt.subplots(figsize=(20,15))
sns.boxplot(x='Age', y='BMI', data=df)
```

```bash
x = df.drop('Outcome',axis=1)
y = df['Outcome']
```

```bash
from sklearn.model_selection import train_test_split
```

```bash
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
```

## Modeling

```bash
from sklearn.linear_model import LogisticRegression
```

```bash
logmodel = LogisticRegression()
```

```bash
logmodel.fit(x_train,y_train)
```

```bash
predictions = logmodel.predict(x_test)
```

```bash
from sklearn.metrics import classification_report
```

```bash
print(classification_report(y_test,predictions))
```
![Alt text](A2.png)

```bash
from sklearn.metrics import confusion_matrix
```

```bash
confusion_matrix(y_test,predictions)
```

```bash
df.head()
```

```bash
print(x)
```
### Membuat feature

Setelah seluruh kolom bertipe data integer dan memiliki nama yang cukup rapih, maka kita dapat membuat fitur dari kolom-kolom tersebut.
Feature digunakan sebagai parameter menghitung hasil estimasi/prediksi yang diharapkan. Hasil estimasi di dataset ini adalah kolom untuk membuat feature dan target dengan codingan sebgai berikut:

```bash
x = pd.DataFrame(df, columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y = df['Outcome'].values.reshape(-1,1)
```

```bash
x_train, x_test, y_train,y_test = train_test_split(x,y , test_size = 0.2 , random_state = 0)
```

```bash
lr = LinearRegression()
lr.fit(x_train,y_train)
```

```bash
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

```bash
x.shape
```

```bash
import pickle

filename = "diabetes.sav"
pickle.dump(lr,open(filename,'wb')) 
```

## Deployment

[My Estimation App](https://appediabetes-ggg7suearpfjpeadzmucuf.streamlit.app/).

![Alt text](A1.png)
