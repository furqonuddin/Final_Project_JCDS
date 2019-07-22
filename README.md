# DIABETES PREDICTION

#### LATAR BELAKANG
- Project ini bertujuan untuk memudahkan para spesialis/dokter
untuk mengambil keputusan terkait "Apakah seseorang terindikasi penyakit diabetes atau tidak?"

#### DESKRIPSI PROJECT
- Tema : Prediksi Diabetes
- Project ini bersumber dari "National Institute of Diabetes and Digestive and Kidney Diseases".
- Semua data adalah pasien perempuan yang berumur lebih dari 21 tahun
- [Sumber Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

#### PENGOLAHAN DATA
- Baca Data
```python
df = pd.read_csv('diabetes.csv')
```
- Cek Nilai NaN
![NaN](4.png)
- Cek Tipe data
![tipedata](2.jpg)
- Baca data
![Data](3.jpg)
- Ubah Data 0 menjadi format "NaN"
```python
df = pd.read_csv(
    'diabetes.csv',
    na_values={ 
        'Glucose' : 0,
        'BloodPressure' : 0,
        'SkinThickness' : 0,
        'Insulin' : 0,
        'BMI' : 0
    }
)
```
- Cek Nilai NaN
![Data](1.jpg)
- Menentukan metode pengisian nilai NaN dengan melihat tabel korelasinya:
![Data](5.jpg)
- Dari tabel korelasi tersebut kita dapat melihat hubungan
  - Glucose         => Insulin
  - BloodPressure   => Age
  - SkinThickness   => BMI
  - Insulin         => Glucose
  - BMI             => SkinThickness
- Maka kita dapat mengisi feature yang kosong berdasarkan Nilai Korelasinya
- Karena terdapat data yang kosong pada kolom tujuan maka dicari penggantinya dengan tingkat korelasi diatas 25% menjadi:
  - Glucose         => Outcome
  - Insulin         => Glucose
  - BloodPressure   => Age
  - BMI             => Outcome
  - SkinThickness   => BMI
- Function program untuk mencari nilai rata2 berdasarkan feature tujuan
```python
def fill_byRata2(i_baru,i_by):
    df1 = df.sort_values(by=i_by)
    by = df1[df1[i_baru].isna()][i_by]
    indeks = by.index.values
    kolom = df1.columns.get_loc(i_baru)
    nilai = by.values

    value_na=[]
    nilai_b =[]
    for x in nilai:
        jml = len(df1[df1[i_by].isin([round(x,2)])])
        mean_x = df1[df1[i_by].isin([round(x,2)])].groupby(i_by, as_index=False).mean()[i_baru].values
        if jml >= 2 and pd.notnull(mean_x)==True:
            nilai_b.append(round(x,2))
            value_na.append(round(mean_x[0],2))
        else:
            to_maks = x + 3
            to_minn = x - 3
            
            while x <= to_maks:                
                x += .01
                jml = len(df1[df1[i_by].isin([round(x,2)])])
                mean_x = df1[df1[i_by].isin([round(x,2)])].groupby(i_by, as_index=False).mean()[i_baru].values
                if jml >= 2 and pd.notnull(mean_x)==True:
                    break
    
            while x >= to_minn:
                mean_x = df1[df1[i_by].isin([round(x,2)])].groupby(i_by, as_index=False).mean()[i_baru].values
                jml = len(df1[df1[i_by].isin([round(x,2)])])
                if jml >= 2 and pd.notnull(mean_x)==True:
                    break
                x -= .01
                                
            nilai_b.append(round(x,2))
            if mean_x.size>0:
                value_na.append(round(mean_x[0],2))
            else:
                value_na.append(np.nan)

    m = 0
    for index in indeks:
        df.iloc[index,kolom] = value_na[m]
        m += 1
    return df
```

- Mengisi nilai NaN
```python
fill_byRata2('Glucose','Outcome')
fill_byRata2('Insulin', 'Glucose')
fill_byRata2('BloodPressure', 'Age')
fill_byRata2('BMI', 'Outcome')
fill_byRata2('SkinThickness', 'BMI')
```
- Cek Nilai NaN
![DataNaN](6.jpg)
- Melakukan fungsi "dropna()" pada nilai NaN
- Convert DataFrame to CSV
```python
df.to_csv('diabetes_new.csv', index=False)
```


#### SPLIT TRAIN MODEL
- Tentukan Sumbu X dan Sumbu Y
```python
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
```
- Spliting Data
- Train data by Split data
  - Logistic Regression
  - Random Forest Classifier
  - Decission Tree Classifier
  - SVM
- Hasil score Tes
![Data](7.jpg)
- Hasil Score by Cross Validation
![Data](8.jpg)
- Berdasarkan data tersebut diambil model Logistic Regression untuk model Machine Learning


#### CONVERT TO BINARY
- Menggunakan fungsi joblib
- Masukkan semua data sebagai Training
```python
import pandas as pd
import joblib

df = pd.read_csv('diabetes_new.csv')

x = df.drop(['Outcome'], axis=1)
y = df['Outcome']

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=500)
model.fit(x, y)

joblib.dump(model, 'modelML')
```

#### APLIKASI FLASK
- Membuat 3 buah HTML
  - home.html
  - hasil.html
  - error.html
- Tampilan Awal Web:
![Data](9.jpg)
- Tampilan Hasil:
![Data](10.jpg)
- Tampilan Error:
![Data](11.jpg)
