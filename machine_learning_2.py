"""




"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import Lasso,LinearRegression, LogisticRegression





iris = load_iris()

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

df_iris["sepal_area"] = df_iris["sepal length (cm)"] * df_iris["sepal width (cm)"]
df_iris["petal_area"] = df_iris["petal length (cm)"] * df_iris["petal width (cm)"]



X=df_iris.drop(["target"], axis=1)
y=df_iris["target"]
X_train_val, X_test ,y_train_val, y_test=train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25,random_state=42,stratify=y_train_val)




# 1.model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_val_knn=knn.predict(X_val)
y_pred_KNN=knn.predict(X_test)

# 2.model
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

logistic = LogisticRegression(max_iter=200)
logistic.fit(X_train_scaled, y_train)
y_pred_val_logistic = logistic.predict(X_val_scaled)
y_pred_logistic = logistic.predict(X_test_scaled)

# 3.model
decision = DecisionTreeClassifier()
decision.fit(X_train_scaled, y_train)

y_pred_val_tree = decision.predict(X_val_scaled)
y_pred_tree = decision.predict(X_test_scaled)

print("--- ACCURACY ---")
print(f"KNN: {accuracy_score(y_val, y_pred_val_knn):.4f}")
print(f"Logistic Regression: {accuracy_score(y_val, y_pred_val_logistic):.4f}")
print(f"Decision Tree: {accuracy_score(y_val, y_pred_val_tree):.4f}")

print("\n--- F1 SCORE ---")
print(f"KNN: {f1_score(y_val, y_pred_val_knn, average='weighted'):.4f}")
print(f"Logistic Regression: {f1_score(y_val, y_pred_val_logistic, average='weighted'):.4f}")
print(f"Decision Tree: {f1_score(y_val, y_pred_val_tree, average='weighted'):.4f}")

print("\n--- PRECISION ---")
print(f"KNN: {precision_score(y_val, y_pred_val_knn, average='weighted'):.4f}")
print(f"Logistic Regression: {precision_score(y_val, y_pred_val_logistic, average='weighted'):.4f}")
print(f"Decision Tree: {precision_score(y_val, y_pred_val_tree, average='weighted'):.4f}")

print("\n--- RECALL ---")
print(f"KNN: {recall_score(y_val, y_pred_val_knn, average='weighted'):.4f}")
print(f"Logistic Regression: {recall_score(y_val, y_pred_val_logistic, average='weighted'):.4f}")
print(f"Decision Tree: {recall_score(y_val, y_pred_val_tree, average='weighted'):.4f}")


"""
--- ACCURACY ---
KNN: 0.9655
Logistic Regression: 0.9655
Decision Tree: 0.9310

--- F1 SCORE ---
KNN: 0.9654
Logistic Regression: 0.9654
Decision Tree: 0.9303

--- PRECISION ---
KNN: 0.9687
Logistic Regression: 0.9687
Decision Tree: 0.9425

--- RECALL ---
KNN: 0.9655
Logistic Regression: 0.9655
Decision Tree: 0.9310


Veri setimize en uygun modeller logistic regression ve KNN şeklinde yorumlayabiliriz ben kullanım kolaylığından dolayı KNN ile devam edeceğim KNN için bir hiperparametre ayarlaması yapalım

"""

grid_params = {"n_neighbors": [3, 5, 7, 10]}

grid = GridSearchCV(
    estimator= KNeighborsClassifier(),
    param_grid=grid_params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train_scaled, y_train)

print(f"\nEn iyi K değeri: {grid.best_params_['n_neighbors']}")

y_pred_grid = grid.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred_grid):.4f}")
print(f"F1: {f1_score(y_test, y_pred_grid, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_grid, average='weighted'):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_grid, average='weighted'):.4f}")

"""
En iyi K değeri: 3
Accuracy: 0.9333
F1: 0.9327
Recall: 0.9333
Precision: 0.9444

Bizde k değeri için 3 kullanmıştık yine aynı setimden devam edeceğim
"""

