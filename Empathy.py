#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
warnings.filterwarnings('ignore')


# # Importing the dataset

# In[2]:


data_path = 'Data Science/Data/EyeT'


# In[3]:


data_files = [csv for csv in os.listdir(data_path) if csv.endswith('.csv')]
len(data_files)


# In[4]:


data = pd.concat([pd.read_csv(os.path.join(data_path, i)) for i in data_files], ignore_index=True)


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


# Check for null values.
for i in data.columns:
    print(i,":",data[i].isnull().sum()/data.shape[0]*100)


# In[8]:


#Obtain a range of different metrics about your numerical columns
data.describe()


# In[9]:


data.hist(figsize=(25, 15)); #Histograms


# ## Data Preprocessing

# In[10]:


#Drop Unnessessary columns
#Preprocesses eye-tracking data by dropping unnecessary columns
unnessessaryColumns =['Timeline name', 'Export date', 
                      'Recording date UTC', 'Mouse position X', 'Recording start time', 
                      'Recording Fixation filter name', 'Presented Stimulus name', 
                      'Recording software version', 'Original Media height', 'Presented Media width', 
                      'Presented Media name', 'Recording date',
                      'Recording duration', 'Event value', 'Sensor', 'Recording name', 
                      'Eye movement type index', 'Recording resolution width', 'Recording resolution height', 
                       'Recording start time UTC', 'Original Media width', 
                      'Presented Media position X (DACSpx)', 'Unnamed: 0', 'Event', 'Presented Media position Y (DACSpx)', 
                      'Mouse position Y', 'Recording monitor latency', 'Project name', 'Presented Media height']

# drop the columns
data_preprocessed = data.drop(columns=unnessessaryColumns)


# In[11]:


# replacing all commas to dots in the number values
data_preprocessed = data_preprocessed.replace(to_replace=r',', value='.', regex=True)

cols_to_modify = ['Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)', 
                          'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)', 'Gaze point left X (MCSnorm)', 
                          'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)','Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z', 'Gaze direction right X', 
                          'Gaze direction right Y', 'Gaze direction right Z',  
                          'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)', 
                          'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)', 
                          'Gaze point left X (DACSmm)', 'Gaze point right Y (MCSnorm)', 
                          'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)']
    
   
data_preprocessed[cols_to_modify] = data_preprocessed[cols_to_modify].astype(float)


# In[12]:


data_preprocessed.info()


# In[13]:


from sklearn.preprocessing import LabelEncoder
convert_Columns = ['Validity left','Validity right','Eye movement type','Pupil diameter left','Pupil diameter right']

# Create a LabelEncoder object
LabelEncoder = LabelEncoder()

# Apply label encoding to 'convert_Columns' list
for i in convert_Columns:
    data_preprocessed[i] = LabelEncoder.fit_transform(data_preprocessed[i])


# In[14]:


data_preprocessed.info()


# In[15]:


# filling remaining NaN values with forward fill method
data_preprocessed = data_preprocessed.fillna(method='ffill')


# In[16]:


# Check for null values.
for i in data_preprocessed.columns:
    print(i,":",data_preprocessed[i].isnull().sum()/data_preprocessed.shape[0]*100)


# In[17]:


data_preprocessed = data_preprocessed.fillna(method='bfill')


# In[18]:


# Check for null values.
for i in data_preprocessed.columns:
    print(i,":",data_preprocessed[i].isnull().sum()/data_preprocessed.shape[0]*100)


# ## Questionnaire Data

# In[19]:


question_Dataset = pd.read_csv("CE888/Questionnaire_datasetIB.csv", encoding= 'unicode_escape')
question_Dataset.head()


# In[20]:


data_preprocessed['Participant name'] = data_preprocessed['Participant name'].str[-2:].astype(int)
data_preprocessed.rename(columns={'Participant name': 'Participant nr'}, inplace=True)


# In[21]:


#Merging with score dataset 
data_preprocessed = data_preprocessed.merge(question_Dataset[['Participant nr','Total Score extended']],on = 'Participant nr',how ='inner')


# In[22]:


data_preprocessed.head()


# # Visualisation

# In[48]:


# Visualizing number of Eye movement types among Participants

# Set the figure size
plt.figure(figsize=(10, 8))

# Create the countplot
ax = sns.countplot(x='Participant name', hue='Eye movement type', data=data, palette='Set1')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# Add a title and axis labels
plt.title('Eye Movement Type by Participant')
plt.xlabel('Participant Name')
plt.ylabel('Count')

# Display the plot
plt.show()


# In[50]:


# ploting the correlation heatmap
corr = data_preprocessed.corr()
sns.set(style='white')
plt.figure(figsize=(30,30))
sns.heatmap(data_preprocessed.drop('Total Score extended', axis=1).corr(), annot=True, cmap='YlGnBu')
# show the plot
plt.show()


# In[39]:


# Grouping the questionnaire data by Participant nr and summing their scores
participant_scores = question_Dataset.groupby('Participant nr')['Total Score extended'].sum().reset_index()

# Plotting the bar graph
plt.figure(figsize=(15, 6))
plt.bar(participant_scores['Participant nr'], participant_scores['Total Score extended'])
plt.title('Total Scores by Participant')
plt.xlabel('Participant Number')
plt.ylabel('Total Score')
plt.xticks(participant_scores['Participant nr'])
plt.grid(axis='y', linestyle='--')
plt.show()


# ## Train-Test Split

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X = data_preprocessed.drop(columns=['Total Score extended'])
y = data_preprocessed['Total Score extended']
#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[25]:


print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[24]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error

RandomForestRegressor = RandomForestRegressor(n_estimators=50, random_state=42)

# training model to the training data
RandomForestRegressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = RandomForestRegressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")
print(f"Mean Absolute Error: {mae}")


# In[ ]:




