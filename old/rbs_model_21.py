#%% Dependencies
###############################################################

from imblearn.base import SamplerMixin
from numpy.lib.histograms import histogram_bin_edges
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score

%matplotlib qt
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plotConfMatrix

from xgboost import XGBClassifier, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# %% Load data
#################################################

data0 = pd.read_excel('data/rbs.xlsx',header=[0,1])

# %% Features
###############################################

features = [
    ('RB', 'Player'),
    ('RB', 'Draft Year'),
    ('RB', 'Conf'),
    ('RB', 'DP'),
    ('Career Average', 'SCRIM Yards/GP'),
    ('Career Average', 'SCRIM TDs/GP'),
    ('Career Best', 'SCRIM Yards/GP'),
    ('Career Best', 'SCRIM TDs/GP'),
    ('Career Last', 'SCRIM Yards/GP'),
    ('Career Last', 'SCRIM TDs/GP'),
    ('College Dominator', 'SCRIM CD'),
    ('Breakout Age', 'RB BOA'),
    ('NFL Fantasy Stats & Finishes', 'Top 24 RB')
]

median_cols = [
    ('Career Average', 'SCRIM Yards/GP'),
    ('Career Average', 'SCRIM TDs/GP'),
    ('Career Best', 'SCRIM Yards/GP'),
    ('Career Best', 'SCRIM TDs/GP'),
    ('Career Last', 'SCRIM Yards/GP'),
    ('Career Last', 'SCRIM TDs/GP'),
    ('College Dominator', 'SCRIM CD'),
]

max_cols = [
    ('Breakout Age', 'RB BOA')
]

data = data0[features].copy()

data[('RB', 'DP')].replace('UDFA',300,inplace=True)

for col in median_cols:
    nums = [n for n in data[col].values if type(n) != str]
    median = np.median(nums)
    data[col].replace('-',median,inplace=True)
    data[col].fillna((data[col].mean()), inplace=True)

for col in max_cols:
    nums = [n for n in data[col].values if type(n) != str]
    max1 = np.max(nums)+2
    data[col].replace('-',max1,inplace=True)
    data[col].fillna((data[col].median()), inplace=True)

data[('NFL Fantasy Stats & Finishes', 'Top 24 RB')].replace('-',0,inplace=True)

rank=[]
for n in data[('NFL Fantasy Stats & Finishes', 'Top 24 RB')].values:
    if n <= 1:
        rank.append(0)
    else:
        rank.append(1)

data.drop([('NFL Fantasy Stats & Finishes', 'Top 24 RB')],1,inplace=True)
data['eval'] = rank


#%%Train and test
rookies = data.loc[data[('RB', 'Draft Year')] == 2021]
# rookies = rookies.loc[rookies[('RB', 'DP')] > 50]
rk_conf = rookies[('RB', 'Conf')]
rk_names = rookies[('RB', 'Player')]
rookies = rookies.iloc[:,3:-1]

features = data.loc[data[('RB', 'Draft Year')] < 2018 ]
features = features.iloc[:,2:]
# features = features.loc[features[('RB', 'DP')] > 50]
X = features.iloc[:,:-1]
y = features.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)

train_conf = X_train[('RB', 'Conf')]
X_train = X_train.iloc[:,1:]
test_conf = X_test[('RB', 'Conf')]
X_test = X_test.iloc[:,1:]

scaler = MinMaxScaler((-1,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_rk = scaler.transform(rookies)

encoder = OneHotEncoder()
encoded_conf_train = encoder.fit_transform(train_conf.values.reshape(-1, 1)).toarray()
encoded_conf_test = encoder.transform(test_conf.values.reshape(-1, 1)).toarray()
X_train = np.concatenate([X_train,encoded_conf_train],axis=1)
X_test = np.concatenate([X_test,encoded_conf_test],axis=1)

encoded_conf_rk = encoder.transform(rk_conf.values.reshape(-1, 1)).toarray()
X_rk = np.concatenate([X_rk,encoded_conf_rk],axis=1)



smote =SMOTE(sampling_strategy='minority',k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)

#%% Plot distributions
################################################
zeros = features.loc[features['eval'] == 0]
ones = features.loc[features['eval'] == 1]
numeric_cols = features.columns[1:-1]
nplot=1
for col in numeric_cols:
    plt.subplot(int(np.ceil(len(numeric_cols)/3)),3,nplot)
    sns.distplot(zeros[col],hist=False,label='Misses')
    sns.distplot(ones[col],hist=False,label='Hits')
    nplot+=1
    plt.legend()
plt.tight_layout()
plt.show()




# %% Modelling
###############################################

estimator = XGBClassifier()
# estimator = LogisticRegression()
param_grid={
            'max_depth':[3,5,12],
            'n_estimators':[50,100,200],
            'objective':['binary:logistic']
            }
# param_grid={}

clf = GridSearchCV(estimator,param_grid,scoring='precision',cv=2,verbose=2,n_jobs=-1)

clf.fit(X_train,y_train)

preds_proba = clf.predict_proba(X_test)[:,1]

preds = clf.predict(X_test)

recall = recall_score(y_test,preds)
precision = precision_score(y_test,preds)
confmat = confusion_matrix(y_test,preds)
plotConfMatrix(confmat,[0,1])

print(f'Precision: {precision}, Recall: {recall}')

# %% Predictions on rookies

rk_preds = clf.predict(X_rk)
rk_probas = clf.predict_proba(X_rk)[:,1]
df = pd.DataFrame([rk_names,rk_preds,rk_probas]).T
df.columns = ['Player Name', 'Hit/Miss Prediction', 'Hit Probability']
df.sort_values('Hit Probability',ascending=False,inplace=True)
df.to_csv('rb_probs.csv')
# %%
