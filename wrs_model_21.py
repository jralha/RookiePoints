#%% Dependencies
###############################################################

from imblearn.base import SamplerMixin
from numpy.lib.histograms import histogram_bin_edges
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score


import matplotlib.pyplot as plt
import seaborn as sns
from utils import plotConfMatrix
%matplotlib qt

from xgboost import XGBClassifier, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# %% Load data
#################################################

data0 = pd.read_excel('data/wrs.xlsx',header=[0,1])

# %% Features
###############################################

features = [
    ('WR', 'Player'),
    ('Draft', 'Draft Year'),
    ('FF_Spaceman', 'Conf'),
    ('Draft', 'DP'),

    # ('Career Best', 'RECs/GP'),
    # ('Career Best', 'REC Yards/GP'),
    # ('Career Best', 'SCRIM TDs/GP'),

    # ('Career Average', 'RECs/GP'),
    # ('Career Average', 'REC Yards/GP'),
    # ('Career Average', 'SCRIM TDs/GP'),

    ('Combine', 'BMI'),
    ('Combine', 'Height'),
    ('Combine', 'Weight'),
    ('Combine', 'Hand Size'),
    ('Combine', 'Arm Length'),
    ('Combine', '40 time'),
    ('Combine', 'Bench'),
    ('Combine', 'Vertical'),
    ('Combine', 'Broad'),
    ('Combine', 'Shuttle'),
    ('Combine', '3 Cone'),

    ('College Dominator', 'REC CD'),

    ('Breakout Age', 'WR BOA (20%)'),
    ('Breakout Age', 'WR BOA (30%)'),

    ('NFL Stats, Finishes, and Milestones', 'Top 24 WR')
]

max_cols = [
    ('Breakout Age', 'WR BOA (20%)'),
    ('Breakout Age', 'WR BOA (30%)'),
]

median_cols = [ f for f in features[3:] if f not in max_cols ]

data = data0[features].copy()

data[('Draft','DP')].replace('UDFA',300,inplace=True)

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

data[('NFL Stats, Finishes, and Milestones', 'Top 24 WR')].replace('-',0,inplace=True)

rank=[]
for n in data[('NFL Stats, Finishes, and Milestones', 'Top 24 WR')].values:
    if n <= 1:
        rank.append(0)
    else:
        rank.append(1)

data.drop([('NFL Stats, Finishes, and Milestones', 'Top 24 WR')],1,inplace=True)
data['eval'] = rank


#%%Train and test
rookies = data.loc[data[('Draft', 'Draft Year')] == 2021]
rk_conf = rookies[('FF_Spaceman', 'Conf')]
rk_names = rookies[('WR', 'Player')]
rookies = rookies.iloc[:,3:-1]

# soph = data.loc[(data[('WR', 'Draft Year')] >= 2018) & (data[('WR', 'Draft Year')] < 2021)]
soph = data.loc[data[('Draft', 'Draft Year')] < 2018 ]
soph_conf = soph[('FF_Spaceman', 'Conf')]
soph_names = soph[('WR', 'Player')]
soph = soph.iloc[:,3:-1]

features = data.loc[data[('Draft', 'Draft Year')] < 2018 ]
features = features.iloc[:,2:]
# features = features.loc[features[('WR', 'DP')] > 32]
X = features.iloc[:,:-1]
y = features.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)

train_conf = X_train[('FF_Spaceman', 'Conf')]
X_train = X_train.iloc[:,1:]
test_conf = X_test[('FF_Spaceman', 'Conf')]
X_test = X_test.iloc[:,1:]

scaler = MinMaxScaler((-1,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_rk = scaler.transform(rookies)
X_soph = scaler.transform(soph)

encoder = OneHotEncoder()
encoded_conf_train = encoder.fit_transform(train_conf.values.reshape(-1, 1)).toarray()
encoded_conf_test = encoder.transform(test_conf.values.reshape(-1, 1)).toarray()
X_train = np.concatenate([X_train,encoded_conf_train],axis=1)
X_test = np.concatenate([X_test,encoded_conf_test],axis=1)

encoded_conf_rk = encoder.transform(rk_conf.values.reshape(-1, 1)).toarray()
X_rk = np.concatenate([X_rk,encoded_conf_rk],axis=1)

encoded_conf_soph = encoder.transform(soph_conf.values.reshape(-1, 1)).toarray()
X_soph = np.concatenate([X_soph,encoded_conf_soph],axis=1)



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

clf = GridSearchCV(estimator,param_grid,scoring='f1',cv=2,verbose=2,n_jobs=-1)

clf.fit(X_train,y_train)
preds_proba = clf.predict_proba(X_test)[:,1]

#%%Threshold
thresholds = range(100)

f1=0
for t in thresholds:
    # preds = clf.predict(X_test)
    threshold = (t+1)/100
    temp_preds = [ 1 if n >= threshold else 0 for n in preds_proba]

    temp_recall = recall_score(y_test,temp_preds,zero_division=0)
    temp_precision = precision_score(y_test,temp_preds,zero_division=0)
    temp_f1 = f1_score(y_test,temp_preds,zero_division=0)
    temp_confmat = confusion_matrix(y_test,temp_preds)

    if temp_f1 >= f1:
        recall = temp_recall
        precision = temp_precision
        confmat = temp_confmat
        preds = temp_preds
        best_threshold = threshold
        f1=temp_f1

plotConfMatrix(confmat,[0,1])

print(f'F1: {f1}, Recall: {recall}, Precision: {precision}')

# %% Predictions on rookies

rk_preds = clf.predict(X_rk)
rk_probas = clf.predict_proba(X_rk)[:,1]
df = pd.DataFrame([rk_names.values,rk_preds,rk_probas]).T
df.columns = ['Player Name', 'Hit/Miss Prediction', 'Hit Probability']
df.sort_values('Hit Probability',ascending=False,inplace=True)
df.reset_index(inplace=True,drop=True)
df.to_csv('wr_probs.csv')

# %% Predictions on players
soph_preds = clf.predict(X_soph)
soph_probas = clf.predict_proba(X_soph)[:,1]
df = pd.DataFrame([soph_names.values,soph_preds,soph_probas]).T
df.columns = ['Player Name', 'Hit/Miss Prediction', 'Hit Probability']
df.sort_values('Hit Probability',ascending=False,inplace=True)
df.reset_index(inplace=True,drop=True)
df.to_csv('wr_probs_18-21.csv')

# %%
