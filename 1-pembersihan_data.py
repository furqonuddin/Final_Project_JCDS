import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# df = pd.read_csv('diabetes.csv')
df = pd.read_csv(
    'diabetes.csv',
    na_values={                                 # untuk merubah nilai kosong => NaN
        'Glucose' : 0,
        'BloodPressure' : 0,
        'SkinThickness' : 0,
        'Insulin' : 0,
        'BMI' : 0
    }
)

print('=====================================================================')
print('Cek Nilai NaN (kosong) pada DataFrame')
print('=====================================================================')
print(df.isnull().sum())

print('\n=====================================================================')
print('Cek Tipe data dari setiap kolom')
print('=====================================================================')
print(df.dtypes)

print('\n=====================================================================')
print('Cek 10 Nilai urutan teratas')
print('=====================================================================')
print(df.head(10))

print('\n=====================================================================')
print('Cek 10 Nilai urutan terbawah')
print('=====================================================================')
print(df.head(10))

# ==============================================================================
'''
1.  Dari data terlihat angka 0 pda setiap kolom mengindikasikan bahwa
    nilai dari kolom tersebut tidak diisi/ bernilai NaN
2.  Ubah nilai dari setiap nilai 0 menjadi NaN,
    Kecuali kolom outcome. Karena nilali 0 pada kolom Outcome mengindikasikan bahwa
    orang tersebut tidak terjangkit penyakit diabetes
3.  Menambahkan fungsi "na_values()" pada bagian fungsi "read_csv()"
'''
# ==============================================================================
print('\n=====================================================================')
print('Nama-nama Kolom:')
print('=====================================================================')
print(df.columns.values)

print('\n=====================================================================')
print('Kolom yang memiliki nilai kosong (NaN):')
print('=====================================================================')
ksg = []
jml_k =[]
for i in df.columns.values:
    a = df[i].isna().sum()
    if a > 0:
        jml_k.append(a)
        ksg.append(i)
print(ksg)
print(jml_k)

# ==============================================================================
'''
Untuk mengetahui cara/ metode yang paling baik untuk mengisi kolom/nilai yang kosong(NaN)
yaitu dengan melihat dari tabel korelasi antar "feature"
'''
# ==============================================================================
print('\n=====================================================================')
print('Tabel korelasi antar "Features:"')
print('=====================================================================')
print(df.corr(method='spearman'))
# ==============================================================================
# corr = df.corr()
# corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# 'RdBu_r' & 'BrBG' are other good diverging colormaps


# ==============================================================================
# Korelasi antar Features (Heatmap)
# ==============================================================================

corrmat = df.corr(method='spearman') 
plt.subplots(figsize=(9,8)) 
plt.subplots_adjust(bottom=0.23)
sns.heatmap(corrmat, cmap ="YlGnBu", linewidths = 0.1) 
plt.title('Heatmap Korelasi')
plt.xticks(rotation = 45)               # atur rotasi dari value x dan y
plt.yticks(rotation = 45)
# plt.tight_layout()
# plt.show()

# ==============================================================================
# from pandas.plotting import scatter_matrix
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(25, 25), diagonal='hist')
plt.show()

# ==============================================================================
'''
1.  Kolom yang terdapat nilai NaN adalah:
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
2.  Dari tabel Korelasi antar "Features" dan Heatmap diagram, 
    kita dapat menentukan metode korelasi yang tepat untuk mengisi nilai yang kosong
3.  Berikut hubungan yang paling baik (berdasarkan nilai corr) antar feature:
    Glucose         => Insulin
    BloodPressure   => Age
    SkinThickness   => BMI
    Insulin         => Glucose
    BMI             => SkinThickness
4.  Membuat functions untuk mengisi kolom yang kosong berdasarkan data diatas 
5.  Melakukan pembualatan nilai dari beberapa feature yang memiliki tipe data float:
    ['Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI']
'''
# ==============================================================================
# Functions Pembulatan nilai
def pembulatan(x):
    br = []
    for i in df[x].values:
        a = round(i,2)
        br.append(a)
    df[x] = br

pembulatan('Glucose')
pembulatan('BloodPressure')
pembulatan('SkinThickness')
pembulatan('Insulin')
pembulatan('BMI')

# ==============================================================================
# Functions fillna

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
        
    # a1 = nilai
    # b1 = np.array(nilai_b)
    # print(len(a1))
    # print(len(b1))
    # print(a1-b1)

    # input nilai NaN
    m = 0
    for index in indeks:
        df.iloc[index,kolom] = value_na[m]
        m += 1

    return df
# ==============================================================================
    
# ===================================
# i_baru          => i_by
# ===================================
# Glucose         => Insulin
# BloodPressure   => Age
# SkinThickness   => BMI
# Insulin         => Glucose
# BMI             => SkinThickness

# ==============================================================================
'''
1.  Melakukan pengecekan apakah pada saat Features "i_baru" bernilai NaN,
    features "i_by" juga bernilai NaN??
2.  Dari data tersebut jika terdapat data pada features yang bernilai NaN,
    maka dicari alternatif korelasi features lain (i_by) yang memiliki nilai
    korelasi diatas 0,25/25%
'''
# ==============================================================================
print('\n=====================================================================')
print('Pengecekan Features yang bernilai NaN:')
print('=====================================================================')
# print(df[df['Glucose'].isna()]['Insulin'])

# ==============================================================================
# ===================================
# i_baru          => i_by
# ===================================
# Glucose         => Outcome
# Insulin         => Glucose
# BloodPressure   => Age
# BMI             => Outcome
# SkinThickness   => BMI

# ==============================================================================
# Fill NaN
fill_byRata2('Glucose','Outcome')
fill_byRata2('Insulin', 'Glucose')
fill_byRata2('BloodPressure', 'Age')
fill_byRata2('BMI', 'Outcome')
fill_byRata2('SkinThickness', 'BMI')


print('\n=====================================================================')
print('Cek Nilai NaN (kosong) pada DataFrame setelah Fillna')
print('=====================================================================')
print(df.isna().sum())

# ==============================================================================
'''
Pada features Insulin masih terdapat nilai yang kosong (NaN)
Maka dilakukan penghapusan nilai dengan menggunakan "dropna()"
'''
# ==============================================================================
# Penghapusan nilai NaN
df = df.dropna()

# ==============================================================================

print('\n=====================================================================')
print('Cek Nilai NaN (kosong) pada DataFrame setelah drop NaN')
print('=====================================================================')
print(df.isna().sum())

# ==============================================================================
# Convert DataFrame to CSV file

df.to_csv('diabetes_new.csv', index=False)