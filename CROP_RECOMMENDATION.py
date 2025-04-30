#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np


# In[63]:


crop = pd.read_csv("Crop_recommendation.csv")


# In[64]:


crop.head()


# In[65]:


crop.shape


# In[66]:


crop.info()


# In[67]:


crop.isnull().sum()


# In[68]:


crop.duplicated().sum()


# In[69]:


crop.describe()


# In[70]:


crop.select_dtypes(include=['number']).corr()


# In[71]:


import seaborn as sns
import matplotlib.pyplot as plt

# Drop non-numeric columns
numeric_crop = crop.drop(columns=['label'])  # adjust column name if it's not 'label'

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_crop.corr(), annot=True, cbar=True, cmap='YlGnBu')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[72]:


crop.label.value_counts()


# In[73]:


crop['label'].unique().size


# In[74]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(crop['P'], kde=True, color='green')  # kde=True shows the curve like distplot
plt.title("Distribution of Phosphorous (P)")
plt.xlabel("Phosphorous")
plt.ylabel("Frequency")
plt.show()


# In[75]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(crop['N'], kde=True, color='blue')  # kde=True adds the curve
plt.title("Distribution of Nitrogen (N)")
plt.xlabel("Nitrogen")
plt.ylabel("Frequency")
plt.show()


# In[76]:


crop.label.unique()


# In[77]:


crop_dict={
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}

crop['label'] = crop['label'].map(crop_dict)


# In[78]:


crop.head()


# In[79]:


crop.label.unique()


# In[80]:


crop.label.value_counts()


# In[81]:


X=crop.drop('label', axis = 1)
y=crop['label']


# In[82]:


X.head()


# In[83]:


y.head()


# In[84]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[85]:


X_train.shape


# In[97]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Ensure X_train and X_test are numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Initialize the scaler
mx = MinMaxScaler()

# Fit and transform on X_train (using numpy arrays)
X_train_scaled = mx.fit_transform(X_train)

# Transform X_test using the same scaler (using numpy arrays)
X_test_scaled = mx.transform(X_test)


# In[98]:


X_train


# In[99]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score


# In[100]:


models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB':GaussianNB(),
    'SVC':SVC(),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'ExtraTreeClassifier':ExtraTreeClassifier(),
    'RandomForestClassifier':RandomForestClassifier(),
    'BaggingClassifier':BaggingClassifier(),
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'AdaBoostClassifier':AdaBoostClassifier()
}


# In[101]:


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{name} model with accuracy: {score}")


# In[102]:


randclf = RandomForestClassifier()
randclf.fit(X_train, y_train)
y_pred = randclf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[103]:


crop.columns


# In[104]:


def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)        # Use transform, not fit_transform
    sc_mx_features = sc.transform(mx_features)  # Same here
    prediction = randclf.predict(sc_mx_features)
    return int(prediction[0])  # Ensure it's a plain int, not array


# In[105]:


crop.head()


# In[106]:


N=90
P= 42
K= 43
temperature= 20.879744
humidity=82.002744
ph=6.502985
rainfall=202.935536

# Your reverse dictionary
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# Make a prediction
predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)

# Print the result
print("Recommended Crop:", reverse_crop_dict[predict])


# In[108]:


N=74
P= 54
K= 38
temperature= 25.65553461
humidity=83.47021081
ph=7.120272972
rainfall=217.3788583

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended Crop:", reverse_crop_dict[int(predict)])


# In[109]:


N = 25
P = 25
K = 30
temperature = 22.0
humidity = 70.0
ph = 6.1
rainfall = 80.0

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended Crop:", reverse_crop_dict[int(predict)])


# In[110]:


N = 55
P = 60
K = 30
temperature = 24.5
humidity = 75.0
ph = 6.7
rainfall = 120.0

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended Crop:", reverse_crop_dict[int(predict)])


# In[111]:


N = 10
P = 80
K = 10
temperature = 9.0
humidity = 82.0
ph = 6.5
rainfall = 100

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended Crop:", reverse_crop_dict[int(predict)])


# In[112]:


N = 20
P = 25
K = 30
temperature = 22.0
humidity = 70.0
ph = 6.1
rainfall = 80.0

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended Crop:", reverse_crop_dict[int(predict)])


# In[113]:


N = 40
P = 40
K = 50
temperature = 30.0
humidity = 65.0
ph = 7.2
rainfall = 105.0

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended Crop:", reverse_crop_dict[int(predict)])


# In[114]:


N = 60
P = 45
K = 50
temperature = 25.5
humidity = 70.0
ph = 7.5
rainfall = 100.0

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended Crop:", reverse_crop_dict[int(predict)])


# In[107]:


N = 10
P = 70
K = 60
temperature = 20.5
humidity = 100.0
ph = 6.5
rainfall = 150.0

predict = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print("Recommended Crop:", reverse_crop_dict[int(predict)])


# In[53]:


import pickle
pickle.dump(randclf, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

