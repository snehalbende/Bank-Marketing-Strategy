#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries for reading and manipulating data
import pandas as pd
import numpy as np
#importing libraries necessary for visulization
import seaborn as sns
import matplotlib.pyplot as plt
# importing libraries for modeling and checking accuracy
from sklearn.model_selection import train_test_split #split
from sklearn.metrics import accuracy_score #metrics
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


# reading the data
data = pd.read_csv("bank.csv")


# In[3]:


data.head()


# In[4]:


data.describe(include = 'all')


# In[5]:


#correlation plot
sns.heatmap(data.corr(),annot =True)


# In[6]:


# visiulizing target value count
sns.countplot(x= 'deposit', data = data , palette="Set3").set_title(" Term Depositors Yes or No?")


# In[7]:


#fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 15))
fig, axs = plt.subplots(ncols=3,nrows = 2 , figsize =(15,10))
sns.distplot(data['balance'], kde = 'False' , color = 'red' , ax = axs[0][0])
sns.distplot(data['pdays'], kde = 'False' , color = 'red', ax = axs[0][1])
sns.distplot(data['campaign'], kde = 'False' , color = 'red', ax = axs[0][2])
sns.distplot(data['previous'], kde = 'False' , color = 'red', ax = axs[1][0])
sns.distplot(data['age'], kde = 'False' , color = 'red', ax = axs[1][1])
sns.distplot(data['duration'], kde = 'False' , color = 'red', ax = axs[1][2])


# In[8]:


plt.figure(figsize=(12, 6))
sns.violinplot(data=data, x="education", y="balance", hue="deposit", palette="RdBu_r")


# In[9]:


plt.figure(figsize=(16, 6))
sns.countplot(x="job", data=data,  order = data['job'].value_counts().index)


# In[10]:


data = data.drop(data.loc[data["job"] == "unknown"].index)


# In[11]:


data = data.drop(data.loc[data["education"] == "unknown"].index)


# In[12]:


sns.countplot(x="marital", hue="deposit", data=data)


# In[13]:


sns.countplot(x="poutcome", color ='red', data=data)


# In[14]:


# since maximum data is in unknown outcome we will delete opoutcome coloumn 
data = data.drop(['poutcome'], axis = 1)


# In[15]:


sns.countplot(x = 'default' , data = data)


# In[16]:


fig, axs = plt.subplots(ncols=3, figsize =(15,8))
sns.countplot(data['housing'] , hue = data['deposit'],color = 'red' , ax = axs[0])
sns.countplot(data['loan'], color = 'blue',hue = data['deposit'], ax = axs[1])
sns.countplot(data['contact'], color = 'green',hue = data['deposit'], ax = axs[2])


# In[17]:


# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
data['deposit']= le.fit_transform(data['deposit']) 
data['default']= le.fit_transform(data['default']) 
data['housing']= le.fit_transform(data['housing']) 
data['loan']= le.fit_transform(data['loan']) 


# In[18]:


data.head()


# In[19]:


df= pd.get_dummies(data, columns=['job', 'marital','education','contact','month'], drop_first=True)


# In[20]:


df.head()


# In[21]:


df.describe()


# In[22]:


pd.set_option('display.max_columns', None)
df.head()


# In[23]:


plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(),annot = True)


# In[24]:


df = df.drop(['contact_unknown'], axis = 1)


# In[25]:


X = df.drop(['deposit'],axis =1)
y = df['deposit']


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state = 42)


# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[28]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression(solver='lbfgs',random_state= 42)
log.fit(X_train,y_train)
pred_log = log.predict(X_test)
print(confusion_matrix(y_test, pred_log))
print(classification_report(y_test, pred_log))


# In[29]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# defining a function to visualize ROC curve
def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[30]:


# visualizing the performace of logistic regression using ROC curve 
from sklearn.metrics import roc_curve, roc_auc_score
probs = log.predict_proba(X_test)  
probs = probs[:, 1]  
fper, tper, thresholds = roc_curve(y_test, probs) 
plot_roc_cur(fper, tper)


# In[31]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[32]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[33]:


probs1 = classifier.predict_proba(X_test)  
probs1 = probs1[:, 1]  
fper, tper, thresholds = roc_curve(y_test, probs1) 
plot_roc_cur(fper, tper)


# In[34]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import decomposition, datasets
# Creating an standardscaler object
std_slc = StandardScaler()

# Creating a pca object
pca = decomposition.PCA()
pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', classifier)])
criterion = ['gini', 'entropy']
max_depth = [2,4,6,8,10,12]
n_components = list(range(1,X.shape[1]+1,1))
parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)
clf_GS = GridSearchCV(pipe, parameters)
clf_GS.fit(X, y)


# In[35]:


classifier1 = DecisionTreeClassifier(max_depth = 8 , criterion = 'gini')
classifier1.fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))


# In[36]:


print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(); print(clf_GS.best_estimator_.get_params()['dec_tree'])


# In[37]:


probs2 = classifier1.predict_proba(X_test)  
probs2 = probs2[:, 1]  
fper, tper, thresholds = roc_curve(y_test, probs2) 
plot_roc_cur(fper, tper)


# In[43]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')
# determinibg optimal k value using elbow method
err = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred = knn.predict(X_test)
    err.append(np.mean(pred != y_test))


# In[44]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), err, color='black', linestyle='dashed', marker='o',
         markerfacecolor='grey', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[45]:


classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

