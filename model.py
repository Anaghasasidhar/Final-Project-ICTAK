#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[2]:


data = pd.read_csv('lending_club_loan_two.csv')


# In[3]:


data.dropna(inplace=True)


# In[4]:


# Typecasting
data['issue_d'] = pd.to_datetime(data['issue_d']).dt.date
data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line']).dt.year
data['emp_length']=data['emp_length'].replace(['10+ years','< 1 year'],['11 years','0 years'])
data['term']=data.term.str.replace(' months','').astype(int)


# In[5]:


def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])


# In[6]:


data['emp_length']=data['emp_length'].apply(emp_length_to_int)


# In[7]:


# Outlier removal
from scipy import stats


# In[8]:


IQR =stats.iqr(data.loan_amnt,interpolation='midpoint')
Q1=data.loan_amnt.quantile(0.25)
Q3=data.loan_amnt.quantile(0.75)
min_limit=Q1-1.5*IQR
max_limit=Q3+1.5*IQR
min_limit,max_limit
data.loc[data['loan_amnt']>max_limit,'loan_amnt']=np.median(data.loan_amnt)


# In[9]:


IQR =stats.iqr(data.int_rate,interpolation='midpoint')
Q1=data.int_rate.quantile(0.25)
Q3=data.int_rate.quantile(0.75)
min_limit=Q1-1.5*IQR
max_limit=Q3+1.5*IQR
min_limit,max_limit
data.loc[data['int_rate']>max_limit,'int_rate']=np.median(data.int_rate)


# In[10]:


IQR =stats.iqr(data.dti,interpolation='midpoint')
Q1=data.dti.quantile(0.25)
Q3=data.dti.quantile(0.75)
min_limit=Q1-1.5*IQR
max_limit=Q3+1.5*IQR
min_limit,max_limit
data.loc[data['dti']>max_limit,'dti']=np.median(data.dti)


# In[11]:


IQR =stats.iqr(data.open_acc,interpolation='midpoint')
Q1=data.open_acc.quantile(0.25)
Q3=data.open_acc.quantile(0.75)
min_limit=Q1-1.5*IQR
max_limit=Q3+1.5*IQR
min_limit,max_limit
data.loc[data['open_acc']>max_limit,'open_acc']=np.median(data.open_acc)


# In[12]:


IQR =stats.iqr(data.mort_acc,interpolation='midpoint')
Q1=data.mort_acc.quantile(0.25)
Q3=data.mort_acc.quantile(0.75)
min_limit=Q1-1.5*IQR
max_limit=Q3+1.5*IQR
min_limit,max_limit
data.loc[data['mort_acc']>max_limit,'mort_acc']=np.median(data.mort_acc)


# In[13]:


IQR =stats.iqr(data.total_acc,interpolation='midpoint')
Q1=data.total_acc.quantile(0.25)
Q3=data.total_acc.quantile(0.75)
min_limit=Q1-1.5*IQR
max_limit=Q3+1.5*IQR
min_limit,max_limit
data.loc[data['total_acc']>max_limit,'total_acc']=np.median(data.total_acc)


# In[14]:


cat_data=data.select_dtypes(include=['object'])
num_data=num_data=data.select_dtypes(include=['float64'])


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


le=LabelEncoder()
data['emp_title']=le.fit_transform(data['emp_title'])
data['home_ownership']=le.fit_transform(data['home_ownership'])
data['verification_status']=le.fit_transform(data['verification_status'])
data['purpose']=le.fit_transform(data['purpose'])
data['application_type']=le.fit_transform(data['application_type']) 
data['initial_list_status']=le.fit_transform(data['initial_list_status'])
data['sub_grade']=le.fit_transform(data['sub_grade'])


# In[17]:


data['loan_status'] = data['loan_status'].map({"Fully Paid": 1, "Charged Off": 0})


# In[18]:


data['loan_status'].value_counts()


# In[19]:


data.drop(data[['address','title','issue_d','emp_title','emp_length','purpose']],axis=1, inplace=True)


# In[20]:


#droping highly correlated features -- instsallment, grade, mort_acc
data.drop(data[['grade']],axis=1, inplace=True)


# In[21]:


data.columns


# In[22]:


# split data into feature and target
x = data[['loan_amnt', 'int_rate', 'annual_inc','dti','open_acc', 'pub_rec','revol_bal','mort_acc',]]
y = data['loan_status']


# In[23]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


# importing SMOTE module from imblearn library  
# pip install imblearn (if you don't have imblearn in your system)  

print("Before Over Sampling, count of the label '1': {}".format(sum(y_train == 1)))  
print("Before Over Sampling, count of the label '0': {} \n".format(sum(y_train == 0)))  
from imblearn.over_sampling import SMOTE  
sm1 = SMOTE(random_state = 2)  
x_train_res, y_train_res = sm1.fit_resample(x_train, y_train.ravel())  
print('After Over Sampling, the shape of the train_X: {}'.format(x_train_res.shape))  
print('After Over Sampling, the shape of the train_y: {} \n'.format(y_train_res.shape))  
print("After Over Sampling, count of the label '1': {}".format(sum(y_train_res == 1)))  
print("After Over Sampling, count of the label '0': {}".format(sum(y_train_res == 0)))


# In[26]:



from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# In[27]:


n_estimators =[int(x) for x in np.linspace(start =10,stop=80,num=10)]
max_features=['auto','sqrt']
max_depth = [2,4]
min_samples_split=[2,5]
min_samples_leaf = [1,2]
bootstrap = [True,False]


# In[28]:


param_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_leaf':min_samples_leaf,
    'bootstrap': bootstrap
}
print(param_grid)


# In[29]:


rf_classifier = RandomForestClassifier()


# In[30]:


from sklearn.model_selection import RandomizedSearchCV
rf_RandomGrid = RandomizedSearchCV(estimator = rf_classifier,param_distributions=param_grid, cv=10 ,verbose=2, n_jobs = 4)
rf_RandomGrid.fit(x_train_res,y_train_res)


# In[31]:


best_model  = rf_RandomGrid.best_estimator_


# In[32]:


y_pred_grid = rf_RandomGrid.predict(x_test)


# In[33]:


import pickle
RF_pkl = open('model.pkl','wb')
pickle.dump(best_model,RF_pkl)
RF_pkl.close()


# In[34]:


le_pkl = open('le.pkl','wb')
pickle.dump(le,le_pkl)
le_pkl.close()


# In[35]:


#with open('scaled_data.pkl','wb') as file:
 #   pickle.dump(scaled_data, file)


# In[36]:


scaler = StandardScaler()
scaler.fit(x_train)

# Save the scaler object using pickle
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


# In[ ]:




