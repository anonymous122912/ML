from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

print("Gaussian Naive Bayes Model accuracy(in %):", metrics.accuracy_score(y_test,y_pred)*100)

cm = confusion_matrix(y_test, y_pred, labels=gnb.classes_)
print("accuracy: ",accuracy_score(y_test, y_pred))
print("confusion matrix: ", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
disp.plot()
plt.show()
df = pd.DataFrame({"Real Values":y_test, "Predicted Values":y_pred})
print("The df is ",df)
print(classification_report(y_test, y_pred))