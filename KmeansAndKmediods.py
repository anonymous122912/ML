import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

x, y=load_iris(return_X_y=True)

kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(x)


kmeans.cluster_centers_


pred = kmeans.fit_predict(x)
pred


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(x[:,0],x[:,1],c=pred,cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center = center[2:4]
    plt.scatter(center[0],center[1],marker='^',c='red')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()
