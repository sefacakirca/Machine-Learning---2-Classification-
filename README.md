# 🌸 Iris Flower Classification

A machine learning classification project using the Iris dataset to predict the species of an Iris flower based on its physical measurements.

## 🎯 Project Objective

The goal of this project is to classify Iris flowers into three different species:

- Setosa
- Versicolor
- Virginica

This is a **multiclass classification** problem.

## 📊 Dataset

The Iris dataset is provided by Scikit-learn and contains:

- 150 samples
- 4 original numerical features
- 3 target classes

### Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target

- `0` → Setosa
- `1` → Versicolor
- `2` → Virginica

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 🔍 Project Workflow

The project follows a standard machine learning workflow:

1. Load and inspect the dataset
2. Check data types and basic statistics
3. Check for missing values
4. Detect and analyze outliers using the IQR method
5. Perform feature engineering
6. Apply feature selection
7. Split the data into train, validation, and test sets
8. Apply scaling where necessary
9. Train multiple classification models
10. Compare validation performance
11. Perform hyperparameter tuning with GridSearchCV
12. Evaluate the final model on the test set

##  Feature Engineering

Two additional features were created:

- `sepal_area` = Sepal Length × Sepal Width
- `petal_area` = Petal Length × Petal Width

These features were created to provide additional information about the physical size of the flower.

##  Models

Three classification algorithms were compared:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree

KNN and Logistic Regression achieved better validation performance than the Decision Tree model.

##  Hyperparameter Tuning

GridSearchCV was used to optimize the `n_neighbors` parameter of the KNN model.

The tested values were:

```text
range(1,21)
