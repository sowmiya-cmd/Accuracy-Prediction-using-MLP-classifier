#Importing Libraries
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
%matplotlib inline
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report,confusion_matrix , accuracy_score ,mean_squared_error
from math import sqrt

#Loading Datset 

 data = pd.read_csv('../input/prediction/fer (1).csv')  
      
    # Printing the dataswet shape 
print ("Dataset Length: ", len(data)) 
print ("Dataset Shape: ", data.shape) 
      
    # Printing the dataset obseravtions 
print ("Dataset: ",data.head(10)) 
 
    # Separating the target variable 
X = data.values[:, 1:12] 
Y = data.values[:, 13]

X, Y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict_proba(X_test[:1])

clf.predict(X_test[:12,:])

print("accuracy :",clf.score(X_test, y_test))
