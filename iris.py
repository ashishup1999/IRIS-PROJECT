# Decision Tree Classification for iris

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('iris.xlsx')
X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, [4]].values

# Dummy code for y set
from sklearn.preprocessing import LabelEncoder
labelenc_y=LabelEncoder()
y=labelenc_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/15, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Reassigning the names to the species
def name(set):
    a=[]
    for i in set:
        if i==0:
            x="setosa"
        elif i==1:
            x="versicolor"
        else:
            x="verginicia"
        a.append(x)
    return a


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_name=name(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Taking input from user and returning the prediction
s_l=float(input("enter the sepal length in c.m. : "))
s_w=float(input("enter the sepal width in c.m. : "))
p_l=float(input("enter the petal length in c.m. : "))
p_w=float(input("enter the petal width in c.m. : "))
y_pred_user=classifier.predict([[s_l,s_w,p_l,p_w]])
print("The predicted species is : ",name(y_pred_user))
