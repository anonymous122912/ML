import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#load the diabetes dataset
diabetes = pd.read_csv('diabetes.csv')
diabetes.head(10)


x = diabetes.drop('Outcome',axis=1)
x
y = diabetes.Outcome
y


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=20)
len(x_test)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = LogisticRegression()
model.fit(x_train,y_train)


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy {:.2f}%".format(accuracy*100))


print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\n Lassification report : \n", classification_report(y_test,y_pred))
