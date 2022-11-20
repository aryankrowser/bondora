#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# To display all the columns of dataframe
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from itertools import product
import seaborn as sns


# In[2]:


df_eng=pd.read_csv('Data_preprocessing.csv',low_memory=False)


# In[3]:


df_eng.dtypes


# In[4]:


df_eng.isna().sum()/len(df_eng)


# In[5]:


df_eng.dtypes


# In[6]:


df_eng.shape


# In[155]:


df_samp=df_eng.sample(n=500)


# In[156]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent', 
                        missing_values=np.nan)
imputer = imputer.fit(df_eng)
df_samp.iloc[:,:] = imputer.transform(df_samp)
df_samp


# In[157]:


df_samp.status.value_counts()


# In[158]:


X = df_samp.copy()
y = X.pop("status")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int


# In[159]:


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)


# In[160]:


mi_scores


# In[161]:


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 8))
plot_mi_scores(mi_scores)


# In[162]:


cols_del=['PreviousEarlyRepaymentsCountBeforeLoan','UseOfLoan','HomeOwnershipType','NoOfPreviousLoansBeforeLoan']


# In[163]:


df_mutual = df_samp.drop(cols_del,axis=1)


# In[164]:


df_mutual.head()


# In[165]:


df_mutual.dtypes


# In[166]:


plt.figure(figsize=(20, 20))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(df_mutual.corr(), dtype=np.bool))
heatmap = sns.heatmap(df_mutual.corr(), mask=mask, vmin=-1, vmax=1, annot=False, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=40);


# In[167]:


df_mutual.dtypes


# In[168]:


df_numerical=df_mutual[['Age','AppliedAmount','Interest','LoanDuration','MonthlyPayment','IncomeTotal','LiabilitiesTotal','AmountOfPreviousLoansBeforeLoan']]


# In[169]:


df_cateogry=df_mutual[['NewCreditCustomer','VerificationType','LanguageCode','Gender','Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','Restructured','CreditScoreEsMicroL','status']]


# In[170]:


df_cateogry.reset_index(inplace=True)


# In[171]:


df_cateogry.drop(['index'],axis=1,inplace=True)


# In[172]:


df_cateogry.head()


# In[173]:


from sklearn import preprocessing
scalar=preprocessing.StandardScaler()
df_numerical_std_1=scalar.fit_transform(df_numerical)


# In[ ]:





# In[174]:


from sklearn.decomposition import PCA

# Create principal components
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df_numerical_std_1)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()


# In[175]:


pca.explained_variance_ratio_


# In[176]:


aftr_pca=pd.concat([df_cateogry,X_pca],axis=1)


# In[177]:


aftr_pca


# In[178]:


categ=aftr_pca[['LanguageCode','Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','CreditScoreEsMicroL']]


# In[179]:


le = preprocessing.LabelEncoder()
aftr_pca[['NewCreditCustomer', 'Restructured']]= aftr_pca[['NewCreditCustomer', 'Restructured']].apply(le.fit_transform)


# In[180]:


aftr_pca[['LanguageCode', 'Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','CreditScoreEsMicroL']]= aftr_pca[['LanguageCode', 'Education','MaritalStatus','EmploymentStatus','EmploymentDurationCurrentEmployer','OccupationArea','CreditScoreEsMicroL']].apply(le.fit_transform)


# In[181]:


aftr_pca.dtypes


# In[182]:


X = aftr_pca.copy()
y = aftr_pca["status"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[183]:


X_train.shape


# In[184]:


from sklearn.ensemble import RandomForestClassifier  
random_classifier= RandomForestClassifier()  
clf = random_classifier.fit(X_train,y_train)
random_pred = clf.predict(X_test)
random_pred_prob = clf.predict_proba(X_test)


# In[185]:


params = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


# In[186]:


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=random_classifier,
                           param_grid=params,
                           cv = 3,
                           n_jobs=-1, verbose=1, scoring="accuracy")


# In[187]:


grid_search.fit(X_train, y_train)


# In[ ]:





# In[ ]:




