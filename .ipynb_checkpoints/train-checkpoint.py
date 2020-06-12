#%% Dependencies
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SVMSMOTE
from sklearn.neural_network import MLPClassifier


# %% Load data, convert data types to numeric and datetime where appropriate
wr_data = pd.read_excel('FF_Spaceman Raw Database Post-Draft 4_30_20.xlsx',sheet_name='WR',header=[1,2])

for col in wr_data.columns:
    if 'DOB' in col:
       wr_data[col] = pd.to_datetime(wr_data[col]) 
    else:
        try:
            wr_data[col] = pd.to_numeric(wr_data[col])
        except:
            continue
#%% Set training and testing features
#Arbitraty, feel free to change those, I didn't perform any feature selection,
#just chose the variables based on intuition.
features=[
    # ('Draft', 'DP'),
    # ('Draft', 'Draft Age'),
    ('FF_Spaceman', 'Conf'),
    ('College Dominator', 'REC CD'),
    ('College Dominator', 'SCRIM CD'),
    # ('Breakout Age', 'WR BOA (20%)'),
    # ('Breakout Age', 'WR BOA (30%)'),
    ('Career Best', 'REC/TM PA'),
    ('Career Best', 'REC Yards/TM PA'),
    ('Career Best', 'SCRIM Yards/GP'),
    ('Career Best', 'SCRIM TDs/GP'),
    ('Career Best', 'SCRIM YDs/ Touch Over TM AVG')


]

#Get all combine stats in one go except HaSS (don't know what this is),
# otherwise does the same as the block above
for col in wr_data['Combine'].columns[1:-1]:
    features.append(('Combine',col))

#NFL production data in one go.
for col in wr_data['NFL Stats, Finishes, and Milestones'].columns:
    features.append(('NFL Stats, Finishes, and Milestones',col))

#%% Split data into Rookies, Veterans and Young Players (Sophs)
# We will train the models on Veteran data to infer Rookie information
# Sophomore data will serve as a sanity check
#This block also separates the training data from NFL production (which is what)
#we are trying to predict.
rooks = wr_data.loc[wr_data[('Draft', 'Draft Year')] == 2020]
vets = wr_data.loc[wr_data[('Draft', 'Draft Year')] < 2014]
soph = wr_data.loc[(wr_data[('Draft', 'Draft Year')] >= 2014) & (wr_data[('Draft', 'Draft Year')] != 2017)]
rook_feats = rooks[features[:-5]]
vet_feats = vets[features[:-5]]
vet_target = vets[features[-5:]]
soph_feats = soph[features[:-5]]
soph_target = soph[features[-5:]]

#Some data tidying up, replacing empty spaces with nans.
#For production and combine data we replace nans with the median value.
#This does introduce a bit of data leakage, but it's pretty safe to assume
#players would be mediocre in combine exercises they chose not to perform in.
#Also, we only have one player with missing college stats.
for feat in features:
    if 'BOA' in feat[1]:
        rook_feats[feat] = rook_feats[feat].replace('-',np.NaN)
        rook_feats[feat] = rook_feats[feat].fillna(value=1+np.nanmax(rook_feats[feat]))
        vet_feats[feat] = vet_feats[feat].replace('-',np.NaN)
        vet_feats[feat] = vet_feats[feat].fillna(value=1+np.nanmax(vet_feats[feat]))
        soph_feats[feat] = soph_feats[feat].replace('-',np.NaN)
        soph_feats[feat] = soph_feats[feat].fillna(value=1+np.nanmax(soph_feats[feat]))
    elif 'Combine' in feat[0] or 'Career' in feat[0]:
        rook_feats[feat] = rook_feats[feat].replace(np.nan,'-').replace('-',np.NaN)
        rook_feats[feat] = rook_feats[feat].fillna(value=np.nanmedian(rook_feats[feat]))
        vet_feats[feat] = vet_feats[feat].replace(np.nan,'-').replace('-',np.NaN)
        vet_feats[feat] = vet_feats[feat].fillna(value=np.nanmedian(vet_feats[feat]))
        soph_feats[feat] = soph_feats[feat].replace(np.nan,'-').replace('-',np.NaN)
        soph_feats[feat] = soph_feats[feat].fillna(value=np.nanmedian(soph_feats[feat]))
    elif 'NFL' in feat[0]:
        vet_target[feat] = vet_target[feat].replace('-',0)
        soph_target[feat] = soph_target[feat].replace('-',0)

#Dropping multilevel column names for simplicity in the next steps
vet_target.columns = vet_target.columns.droplevel()
soph_target.columns = soph_target.columns.droplevel()
vet_feats.columns = vet_feats.columns.droplevel()
soph_feats.columns = soph_feats.columns.droplevel()
rook_feats.columns = rook_feats.columns.droplevel()
# %% Function where we set tiers to each player based on their
# real life fantasy performance in the NFL.
# We also generate a binary classifier where he merely try to define if a player
# was a hit or a miss.
# The cut points are also completely arbitrary.
def set_tiers(df):
    tiers = np.zeros(len(df))
    hit = np.zeros(len(df))
    for i in range(len(tiers)):
        temp = df.iloc[i]
        if temp['Top5'] >= 1.0:
            tiers[i] = 1
            hit[i] = 1
        elif temp['Top12'] > 1.0:
            tiers[i] = 2
            hit[i] = 1
        elif temp['Top12'] == 1.0:
            tiers[i] = 3
            hit[i] = 1
        elif temp['Top24'] >= 1.0:
            tiers[i] = 4
            hit[i] = 1
        elif temp['Top36'] > 1.0:
            tiers[i] = 5
            hit[i] = 0
        else:
            tiers[i] = 6
            hit[i] = 0
    
    return hit, tiers

hit_vets, tiers_vets = set_tiers(vet_target)
hit_soph, tiers_soph = set_tiers(soph_target)


# %% Scalling data to a standard scale, onehot encoding conferences.
# Set X for models
def process_df(feat,encoder):
    scaler = StandardScaler()

    encoded = encoder.transform(feat['Conf'].values.reshape(-1,1)).toarray()
    feat = feat.drop(['Conf'],axis=1)

    feat = feat.replace('UDFA',np.nan).replace(np.nan,256)
    scaled = scaler.fit_transform(feat)

    out = np.concatenate([scaled,encoded],axis=1)

    return out 

encoder = OneHotEncoder()
encoder.fit(wr_data[('FF_Spaceman','Conf')].values.reshape(-1,1))
X = process_df(vet_feats,encoder)
X_rk = process_df(rook_feats,encoder)
X_soph = process_df(soph_feats,encoder)

#%%Define model and sampling
# estimator = xgb.XGBClassifier(objective='multi:softmax',num_class=6)
estimator = MLPClassifier()
parameters = {}
model = GridSearchCV(estimator, parameters,cv=10,n_jobs=-1)
sm = SVMSMOTE(sampling_strategy='all')

#%% Split data for tiers model
def plotPerClassPerformance(metricList,labels,labelCount):
  leftmargin = 1.0 # inches
  rightmargin = 1.0 # inches
  categorysize = 0.5 # inches
  figwidth = leftmargin + rightmargin + (len(labels) * categorysize)
  plt.figure(figsize=(figwidth, figwidth))

  plt.scatter(labelCount,metricList)

  for label, x, y in zip(labels, labelCount, metricList):
    plt.annotate(
        label,
        xy=(x, y), xytext=(10+20*np.random.random(1), 10+20*np.random.random(1)),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
  



model_tiers = model

X_res, y_res = sm.fit_resample(X,tiers_vets) #This is wrong, don't resample before split, I'm doing it for the memes
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, stratify=y_res,random_state=123)

# model_tiers.fit(X,tiers_vets)
preds = model_tiers.predict(X_soph)
cf = confusion_matrix(tiers_soph,preds,normalize='pred')
rec = recall_score(tiers_soph,preds,average=None)
labels, count = np.unique(tiers_soph, return_counts=True)
# plotConfMatrix(cf,labels=range(6))
plotPerClassPerformance(metricList=rec,labels=labels,labelCount=count)


# %% Split data for hit/miss model
model_hit = model

X_res, y_res = sm.fit_resample(X,hit_vets)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, stratify=y_res,random_state=123)

model_hit.fit(X,hit_vets)
preds_h = model_hit.predict(X_soph)
cf = confusion_matrix(hit_soph,preds_h,normalize='true')
plotConfMatrix(cf,labels=range(2))
# %%
pred_soph_tier = model_tiers.predict(X_soph)
pred_soph_hit = model_hit.predict(X_soph)
pred_rook_tier = model_tiers.predict(X_rk)
pred_rook_hit = model_hit.predict(X_rk)

# %%
soph['Pred Tier'] = pred_soph_tier
soph['Pred Hit'] = pred_soph_hit
rooks['Pred Tier'] = pred_rook_tier
rooks['Pred Hit'] = pred_rook_hit

out_feats =    [ ('WR','Player'),
    ('Pred Tier',''),
    ('Pred Hit',''),
    ('Draft', 'DP'),
    ('Draft', 'Draft Age'),
    ('FF_Spaceman', 'Conf'),
    ('College Dominator', 'REC CD'),
    ('College Dominator', 'SCRIM CD'),
    ('Breakout Age', 'WR BOA (20%)'),
    ('Breakout Age', 'WR BOA (30%)'),
    ('Career Best', 'REC/TM PA'),
    ('Career Best', 'REC Yards/TM PA'),
    ('Career Best', 'SCRIM Yards/GP'),
    ('Career Best', 'SCRIM TDs/GP'),
    ('Career Best', 'SCRIM YDs/ Touch Over TM AVG') ]
for col in wr_data['Combine'].columns[:-1]:
    out_feats.append(('Combine',col))

out_soph = soph[out_feats]
out_rk = rooks[out_feats]

out_soph.to_excel('soph.xls')
out_rk.to_excel('rook.xls')

# %%
