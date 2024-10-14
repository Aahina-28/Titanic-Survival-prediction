import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_data = pd.read_csv("Titanic-Dataset.csv")
titanic_data.head()

titanic_data.shape

titanic_data.info()
titanic_data.isnull().sum()

## Handling Missing Value

# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns="Cabin", axis=1)

# replacing the missing values in "Age" column with mean value
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace=True)

# finding the mode value of "Embarked" column
titanic_data["Embarked"].mode()

titanic_data["Embarked"].mode()[0]

# replacing the missing values in "Embarked" column with mode value
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)

# check the number of missing values in each column
titanic_data.isnull().sum()

## Data Analysis
titanic_data.describe()
titanic_data["Survived"].value_counts()

## Data Visualization
sns.countplot(x="Survived", data=titanic_data)

titanic_data["Sex"].value_counts()

sns.countplot(x="Sex", data=titanic_data)

sns.countplot(x="Sex", hue="Survived", data=titanic_data)

# making a count plot for "Pclass" column
sns.countplot(x="Pclass", data=titanic_data)

sns.countplot(x="Pclass", hue="Survived", data=titanic_data)

## Encoding the Categorical Columns
titanic_data["Sex"].value_counts()

titanic_data["Embarked"].value_counts()

titanic_data.replace(
    {"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}},
    inplace=True,
)

titanic_data.head()

## seperating training and test data
X = titanic_data.drop(columns=["PassengerId", "Name", "Ticket", "Survived"], axis=1)
Y = titanic_data["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

X_train.shape, X_test.shape

model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)

X_train_prediction

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy score of training data : ", training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
X_test_prediction

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy score of test data : ", test_data_accuracy)
