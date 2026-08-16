"""




"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

iris = load_iris(return_X_y=True)

df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris["target"] = iris.target
print(df_iris.head())
"""
    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0
"""
"""
Veri setimiz iris veri setidir iris bitkisinin hangi çesidi olduğunu bize açıklar 
Hedef değişkenimiz target değişkenidir bize iris bitkisinin çesidini söyler (Setosa, Versicolour ve Virginica)
Ortada 3 farklı sınıftan bahsettiğimiz için bu bir sınıflandırma problemidir
"""
print(df_iris.shape)
print(df_iris.dtypes)
print(df_iris.info())
"""
Satır ve sutün sayısı

(150, 5)

Veri Tipleri

sepal length (cm)    float64
sepal width (cm)     float64
petal length (cm)    float64
petal width (cm)     float64
target                 int64
dtype: object



Info

RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
 4   target             150 non-null    int64  
dtypes: float64(4), int64(1)
"""

print(df_iris.isnull().sum())
"""
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
target               0
"""
sayisal_sutunlar = df_iris.select_dtypes(include=["int64","float64"])

aykiri_deger_maskesi = pd.Series(False)

for sutun in sayisal_sutunlar:

    q1 = df_iris[sutun].quantile(0.25)
    q3 = df_iris[sutun].quantile(0.75)

    iqr = q3 - q1

    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr

    sutun_maskesi = (
        (df_iris[sutun] < alt_sinir) | (df_iris[sutun] > ust_sinir)
    )

    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi

df_iris=df_iris.loc[~aykiri_deger_maskesi].copy()
df_iris.reset_index(drop=True, inplace=True)


X_train_val, X_test ,y_train_val, y_test=train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,random_state=42,stratify=y_train_val)


