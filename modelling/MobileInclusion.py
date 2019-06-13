
#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()


#%%
train = pd.read_csv("data/training.csv")
test = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")


#%%
print("train shape",train.shape)
print("test shape", test.shape)


#%%
train.head()


#%%
# Handle null values
train.isnull().any(axis=0)


#%%
# data types
train.dtypes


#%%
# data description
train.describe()

#%%
## visualise disttribution of age
plt.hist(train.Q1)
plt.title("Histogram of age")
plt.show()


#%%
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.cluster import k_means

#%%
# drop id column in both train and test set
train.drop('ID', inplace=True, axis = 1)
test.drop('ID',inplace=True, axis=1)

#%%
feature_names = test.columns
features = train[feature_names]
labels = train.mobile_money_classification


#%%
x_tr,x_tes,y_tr,y_tes = train_test_split(features,labels,test_size=0.20,random_state=42)


#%%
# Functions to train the model
def model_train(x_tr,x_tes,y_tr,y_tes, alg):

    alg.fit(x_tr,y_tr)

    pred = alg.predict_proba(x_tes)

    feat_imp = pd.Series(alg.feature_importances_,x_tr.columns).sort_values(ascending=False)
    feat_imp.plot(kind="bar", title = "Feature Importance")
    plt.xlabel("features")
    plt.ylabel("Scores")
    print("Log Loss:", log_loss(y_tes,pred, labels=alg.classes_))
    print()
    return alg

#%%
# Train the model
alg_1 = GradientBoostingClassifier()
print("Grandient Boosting Machine")
model_1 = model_train(features,features,labels,labels,alg_1)
#%%
print("Random Forest")
alg_2 = RandomForestClassifier()
model_2 = model_train(x_tr,x_tes,y_tr,y_tes,alg_2)

#%%
# Making preddictions
predictions = model_1.predict_proba(test)

#%%
#creating submissions
sub = sample_sub
sub[['no_financial_services', 'other_only', 'mm_only', 'mm_plus']] = predictions
sub.to_csv("sumbission1.csv",index=False)

#%%
# create clusters for age
age_clusters = np.array(features.Q1)
n_clusters = 5
clst = k_means(age_clusters.reshape(-1,1),n_clusters=n_clusters)

#%%
# update clusters to the dataset
features['age_clst'] = clst[1]

#%%
# create clusters for longitude
lon = np.array(features.Longitude)
long_clst = k_means(lon.reshape(-1,1), n_clusters=6)