#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# To display all the columns of dataframe
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns


# In[2]:


df_preprocessed=pd.read_csv('Bondora_preprocessed.csv')


# In[3]:


df_preprocessed.head()


# In[4]:


df_preprocessed.shape


# In[5]:


df_preprocessed.dtypes


# In[6]:


df_preprocessed.isnull().sum()


# In[7]:


df_preprocessed.columns


# In[8]:


df_preprocessed["VerificationType"] = df_preprocessed["VerificationType"].replace(np.NaN, df_preprocessed["VerificationType"].mean())
print(df_preprocessed["VerificationType"][:10])


# In[9]:


df_preprocessed["Gender"] = df_preprocessed["Gender"].replace(np.NaN, df_preprocessed["Gender"].mean())
print(df_preprocessed["Gender"][:10])


# In[10]:


df_preprocessed["MonthlyPayment"] = df_preprocessed["MonthlyPayment"].replace(np.NaN, df_preprocessed["MonthlyPayment"].mean())
print(df_preprocessed["MonthlyPayment"][:10])


# In[11]:


X = df_preprocessed.copy()
y = X.pop("Status")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes
discrete_features = X.dtypes == int


# In[12]:


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)


# In[13]:


mi_scores


# In[14]:


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 8))
plot_mi_scores(mi_scores)


# In[15]:


plt.figure(figsize=(20, 20))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df_preprocessed.corr(), dtype=np.bool))
heatmap = sns.heatmap(df_preprocessed.corr(), mask=mask, vmin=-1, vmax=1, annot=False, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=40);


# In[16]:


df_numerical=df_preprocessed[['Age','AppliedAmount','Interest','LoanDuration','MonthlyPayment','IncomeTotal','LiabilitiesTotal','AmountOfPreviousLoansBeforeLoan']]


# In[17]:


df_numerical.shape


# In[18]:


df_categorical=df_preprocessed[['NewCreditCustomer','VerificationType','LanguageCode','Gender','Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','Restructured','CreditScoreEsMicroL','Status']]


# In[19]:


df_categorical.shape


# In[20]:


from sklearn import preprocessing
scalar=preprocessing.StandardScaler()
df_numerical_std_1=scalar.fit_transform(df_numerical)


# In[21]:


from sklearn.decomposition import PCA

# Create principal components
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df_numerical_std_1)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()


# In[22]:


aftr_pca=pd.concat([df_categorical,X_pca],axis=1)


# In[23]:


aftr_pca


# In[24]:


lab_encod = preprocessing.LabelEncoder()
aftr_pca[['NewCreditCustomer', 'Restructured','LanguageCode', 'Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','CreditScoreEsMicroL']]= aftr_pca[['NewCreditCustomer', 'Restructured','LanguageCode', 'Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','CreditScoreEsMicroL']].apply(lab_encod.fit_transform)


# In[25]:


aftr_pca.dtypes


# In[26]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model=ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)


# In[27]:


feat_import=pd.Series(model.feature_importances_,index=X.columns)
feat_import.nlargest(10).plot(kind='barh')


# In[28]:


plt.figure(figsize=(20, 20))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(aftr_pca.corr(), dtype=np.bool))
heatmap = sns.heatmap(aftr_pca.corr(), mask=mask, vmin=-1, vmax=1, annot=False, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=40);


# In[29]:


aftr_pca.columns


# In[30]:


aftr_pca.isnull().sum()


# In[31]:


aftr_pca.head()


# In[32]:


plt.figure(figsize=(20, 16))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(aftr_pca.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# In[33]:


aftr_pca.shape


# In[34]:


X = aftr_pca.copy()
y = aftr_pca["Status"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[35]:


from sklearn.ensemble import RandomForestClassifier  
random_classifier= RandomForestClassifier()  
clf = random_classifier.fit(X_train,y_train)
random_pred = clf.predict(X_test)
random_pred_prob = clf.predict_proba(X_test)


# In[36]:


params = {'bootstrap': [True, False],
 'max_depth': [10, 20, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400]}


# In[37]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=random_classifier,
                           param_grid=params,
                           cv = 2,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[38]:


grid_search.fit(X_train, y_train)


# In[39]:


random_predict_rf = grid_search.predict(X_test)
random_predict_prob = grid_search.predict_proba(X_test)


# In[40]:


y_train.shape


# In[41]:


grid_search.best_score_


# In[42]:


grid_search.best_estimator_


# In[43]:


from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
confusion_matrix(y_test,random_predict_rf)


# In[44]:


# Random forest report

class_report_RF=classification_report(y_test, random_predict_rf)
confusion_matrix(y_test,random_predict_rf)

print(class_report_RF)
print('')
print('')
print('RANDOM FOREST')
print('Accuracy Score -RFM:', metrics.accuracy_score(y_test, random_predict_rf))  
print('F1 Score - RFM:', metrics.f1_score(y_test, random_predict_rf,average = "micro")) 
print('Precision - RFM:', metrics.precision_score(y_test, random_predict_rf, average = "micro"))
print('Recall - RFM:', metrics.recall_score(y_test, random_predict_rf, average = "micro"))


# In[45]:


#logistic regression
from sklearn.linear_model import LogisticRegression
log_classifier=LogisticRegression()


# In[46]:


from sklearn.model_selection import GridSearchCV
parameter_log={'penalty':['l1','l2','elasticnet'],'C':[1,2,3,4,5,6,10,20,30,40,50],'max_iter':[100,200,300]}


# In[47]:


classifier_regressor=GridSearchCV(log_classifier,param_grid=parameter_log,scoring='accuracy',cv=5)
classifier_regressor.fit(X_train,y_train)


# In[48]:


print(classifier_regressor.best_params_)
print(classifier_regressor.best_score_)
y_pred_log=classifier_regressor.predict(X_test)


# In[49]:


score_log=accuracy_score(y_pred_log,y_test)
print(score_log)


# In[50]:


print(classification_report(y_pred_log,y_test))


# In[51]:


confusion_matrix(y_test,y_pred_log)


# In[52]:


# LogisticRegression report

class_report_LR=classification_report(y_test, random_predict_rf)
confusion_matrix(y_test,random_predict_rf)

print(class_report_LR)
print('')
print('')
print('LogisticRegression')
print('Accuracy Score -LR:', metrics.accuracy_score(y_test, y_pred_log))  
print('F1 Score - LR:', metrics.f1_score(y_test, y_pred_log,average = "micro")) 
print('Precision - LR:', metrics.precision_score(y_test, y_pred_log, average = "micro"))
print('Recall - LR:', metrics.recall_score(y_test, y_pred_log, average = "micro"))


# In[53]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[54]:


pipeline_randomforest=Pipeline([('std', StandardScaler()),('pca', PCA(n_components = 5)),('rf_classifier',RandomForestClassifier())])


# In[55]:


pipeline_randomforest.fit(X_train,y_train)


# In[56]:


pred=pipeline_randomforest.predict(X_test)


# In[57]:


accuracy_score(y_test,pred)


# In[58]:


# report of pipeline RF
class_report_RF=classification_report(y_test, pred)
print(class_report_RF)
print('')
print('')
print('RANDOM FOREST MODEL')
print('Accuracy Score -RFM:', metrics.accuracy_score(y_test, pred))  
print('F1 Score - RFM:', metrics.f1_score(y_test, pred,average = "micro")) 
print('Precision - RFM:', metrics.precision_score(y_test, pred, average = "micro"))
print('Recall - RFM:', metrics.recall_score(y_test, pred, average = "macro"))


# In[59]:


pipeline_LogisticRegression=Pipeline([('std', StandardScaler()),('pca', PCA(n_components = 5)),('logistic_regr',LogisticRegression())])


# In[60]:


pipeline_LogisticRegression.fit(X_train,y_train)


# In[61]:


pred_logistic=pipeline_LogisticRegression.predict(X_test)


# In[62]:


accuracy_score(y_test,pred_logistic)


# In[63]:


# report of pipeline LR
class_report_LR=classification_report(y_test, pred_logistic)
print(class_report_LR)
print('')
print('')
print('LOGISTIC REGRESSION MODEL')
print('Accuracy Score -LR:', metrics.accuracy_score(y_test, pred_logistic))  
print('F1 Score - LR:', metrics.f1_score(y_test, pred_logistic,average = "micro")) 
print('Precision - LR:', metrics.precision_score(y_test, pred_logistic, average = "micro"))
print('Recall - LR:', metrics.recall_score(y_test, pred_logistic, average = "macro"))


# In[64]:


aftr_pca.to_csv("credit_pipeline.csv")


# In[ ]:


import pickle
file = open('')

