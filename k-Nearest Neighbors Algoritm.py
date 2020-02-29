#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt #data visualization

from sklearn.datasets import make_blobs #synthetic dataset
from sklearn.neighbors import KNeighborsClassifier #kNN classifier
from sklearn.model_selection import train_test_split #train and test sets


# In[67]:


#create synthetic dataset
X, y = make_blobs(n_samples = 100, n_features = 2, centers = 4,
                       cluster_std = 1.5, random_state = 4)

#scatter plot of dataset
plt.figure(figsize = (10,6))
plt.scatter(X[:,0], X[:,1], c=y, marker= 'o', s=50)
plt.show()


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# In[72]:


knn5 = KNeighborsClassifier() #k=5
knn1 = KNeighborsClassifier(n_neighbors=1) #k=1


# In[73]:


knn5.fit(X_train, y_train)
knn1.fit(X_train, y_train)


# In[74]:


y_pred_5 = knn5.predict(X_test)
y_pred_1 = knn1.predict(X_test)


# In[101]:


from sklearn.metrics import accuracy_score
print("Accuracy of kNN with k=5", accuracy_score(y_test, y_pred_5))
print("Accuracy of kNN with k=1", accuracy_score(y_test, y_pred_1))


# In[106]:


plt.figure(figsize = (10,6))
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker= 'o', s=50)
plt.title("Original data", fontsize=20)
plt.show()


# In[107]:


plt.figure(figsize = (10,6))
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_5, marker= 'o', s=50)
plt.title("Predicted values with k=5", fontsize=20)
plt.show()


# In[108]:


plt.figure(figsize = (10,6))
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_1, marker= 'o', s=50)
plt.title("Predicted values with k=1", fontsize=20)
plt.show()


# ### How to find the best k value

# In[135]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV


# In[120]:


cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)


# In[138]:


knn_grid = GridSearchCV(estimator = KNeighborsClassifier(), 
                        param_grid={'n_neighbors': np.arange(1,20)}, cv=5)


# In[139]:


knn_grid.fit(X_cancer, y_cancer)


# In[140]:


knn_grid.best_params_


# In[ ]:




