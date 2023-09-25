#!/usr/bin/env python
# coding: utf-8

# In[260]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[261]:


data=pd.read_csv('crime_data.csv')


# In[262]:


data


# In[263]:


data.head(7)


# In[264]:


data.shape


#  Act 379-Robbery
# 
#  Act 13-Gambling
# 
#  Act 279-Accident
# 
#  Act 323-Violence
# 
#  Act 302-Murder
# 
#  Act 363-Kidnapping

# In[265]:


data.info()


# In[266]:


data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')


# In[267]:


data['timestamp'] = pd.to_datetime(data['timestamp'], format = '%d/%m/%Y %H:%M:%S')


# In[268]:


data['timestamp']


# In[269]:


# DATE TIME STAMP FUNCTION
column_1 = data.iloc[:,0]

db=pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              "hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              "dayofweek": column_1.dt.dayofweek,
              "weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })


# In[270]:


dataset1=data.drop('timestamp',axis=1)


# In[271]:


data1=pd.concat([db,dataset1],axis=1)
data1


# In[272]:


data1.shape


# ## Data Analysis

# In[273]:


data1.info()


# In[274]:


data1.isnull().sum()


# In[275]:


data1.dropna(inplace=True)


# In[276]:


data1.isnull().sum()


# In[277]:


data1.head()


# In[278]:


data1.columns[1]


# ## Data Visualization & Analysis

# In[279]:


sns.pairplot(data1,hue='act363')


# In[280]:


sns.pairplot(data1,hue='act323')


# In[281]:


df1 = pd.DataFrame(data=data1, columns=['act13', 'act323', 'act379'])


# In[282]:


df1.plot.kde()
plt.show()


# In[283]:


df2 = pd.DataFrame(data=data1, columns=['act13', 'act323', 'act379', 'act279', 'act363', 'act302'])


# In[284]:


df2.plot.kde()
plt.show()


# ## X & Y array

# In[285]:


data1.head()


# In[286]:


data1.shape


# In[287]:


X=data1.iloc[:,[1,2,3,4,6,16,17]].values
#month,day,hour,dayofyear,weekofyear,latitude,longitude


# In[288]:


X[4]


# In[289]:


y=data1.iloc[:,[10,11,12,13,14,15]].values
#act379	act13	act279	act323	act363	act302


# In[290]:


y[4]


# In[291]:


## Splitting the data


# In[292]:


from sklearn.model_selection import train_test_split


# In[293]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)


# In[294]:


X_train.shape


# In[295]:


X_test.shape


# ## Creating & Training KNN Model

# In[296]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)


# In[297]:


acc1=knn.score(X_test,y_test)
acc1


# In[298]:


knn.score(X_train,y_train)


# ## Creating & Training Decision Tree Model

# In[299]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=500, random_state=300)


# In[300]:


dtree.fit(X_train,y_train)


# In[301]:


y_pred=dtree.predict(X_test)


# In[302]:


acc2=dtree.score(X_test,y_test)
acc2


# In[303]:


dtree.score(X_train,y_train)


# In[304]:


y_pred


# ## Creating & Training Random Forest Model

# In[305]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[306]:


y_pred=rfc.predict(X_test)


# In[307]:


X_test[0]


# In[308]:


X_test[10]


# In[309]:


y_pred[10]


# In[310]:


acc3=rfc.score(X_test,y_test)
acc3


# In[311]:


rfc.score(X_train,y_train)


# # Prediction

# In[312]:


test_vector = np.reshape(np.asarray([2., 28., 15., 59. ,  9. , 22.723873,75.828416]),(1,7))
p =np.array(rfc.predict(test_vector)[0])
print(p)

label = ['Robbery','Gambling','Accident','Violence','Kidnapping','Murder']
print (label[3])


# In[313]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('KNN','DecisionTree','RF')
y_pos = np.arange(len(objects))
performance = [acc1,acc2,acc3]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy Level')
plt.title('Accuracy of Algorithms')
 
plt.show()


# In[314]:


#!pip install keras


# In[315]:


#!pip install tensorflow


# ## deep learning model

# In[316]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a random classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM ensemble
n_estimators = 3
max_epochs = 10
n_neurons = 50
ensemble = []

for i in range(n_estimators):
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=max_epochs, verbose=0)
    ensemble.append(model)

# Predict on the testing data using the ensemble
y_preds = []
for model in ensemble:
    y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
    y_preds.append(y_pred)

# Calculate the final predictions as the majority vote of the ensemble
y_final_pred = []
for i in range(len(y_preds[0])):
    votes = [y[i] for y in y_preds]
    y_final_pred.append(max(set(votes), key=votes.count))

# Calculate the accuracy of the ensemble
accuracy = accuracy_score(y_test, y_final_pred)
acc4=(f"Accuracy: {accuracy}")
print(acc4)


# In[ ]:




