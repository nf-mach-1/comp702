'''@Author : Nhlakanipho Fortunatus Machi
@Description : Plant Classification using Leaves Recognition
COMP 702 - Image Processing'''

import os

import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
plt.show()
iris=pd.read_csv('iris.csv')
iris.head()
sns.pairplot(data=iris)
sns.pairplot(data=iris, hue='species', palette='Set2')
from sklearn.model_selection import train_test_split
# Separating the independent variables from dependent variables
x=iris.iloc[:,:-1]
y=iris.iloc[:,4]
x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.30)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train, y_train)
pred=model.predict(x_test)

# Importing the classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test, pred))