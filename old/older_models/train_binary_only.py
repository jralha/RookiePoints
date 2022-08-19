#%% Dependencies
############################################################################################################
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from utils import plotConfMatrix

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

# %% Load data, convert data types to numeric and datetime where appropriate
############################################################################################################
# rb_data = pd.read_excel('FF_Spaceman Raw Database Post-Draft 4_30_20.xlsx',sheet_name='RB',header=[1,2])
wr_data = pd.read_excel('FF_Spaceman Raw Database Post-Draft 4_30_20.xlsx',sheet_name='WR',header=[1,2])

for col in wr_data.columns:
    if 'DOB' in col:
       wr_data[col] = pd.to_datetime(wr_data[col]) 
    #    rb_data[col] = pd.to_datetime(rb_data[col]) 
    else:
        try:
            wr_data[col] = pd.to_numeric(wr_data[col])
            # rb_data[col] = pd.to_numeric(rb_data[col])
        except:
            continue


#%% Set training and testing features
############################################################################################################
#Arbitraty, feel free to change those, I didn't perform any feature selection,
#just chose the variables based on intuition.
features=[
    ('Unnamed: 0_level_0','Player'),
    ('Draft', 'Draft Year'),
    # ('Draft', 'DP'),
    ('FF_Spaceman', 'Conf'),
    ('College Dominator', 'REC CD'),
    ('Breakout Age', 'WR BOA (20%)'),
    ('Career Best', 'REC/TM PA'),
    ('Career Best', 'REC Yards/TM PA'),
    ('Career Best', 'PPR/GP'),
    # ('Career Best', 'SCRIM YDs/ Touch Over TM AVG'),
    # ('Career Average', 'REC/TM PA'),
    # ('Career Average', 'REC Yards/TM PA'),
    # ('Career Average', 'PPR/GP'),
    # ('Career Average', 'SCRIM YDs/ Touch Over TM AVG'),
    # ('Career Last', 'REC/TM PA'),
    # ('Career Last', 'REC Yards/TM PA'),
    # ('Career Last', 'PPR/GP'),
    # ('Career Last', 'SCRIM YDs/ Touch Over TM AVG')


]

#Get all combine stats in one go except HaSS (don't know what this is),
# otherwise does the same as the block above
for col in wr_data['Combine'].columns[1:-1]:
    features.append(('Combine',col))

#NFL production data in one go.
for col in wr_data['NFL Stats, Finishes, and Milestones'].columns:
    features.append(('NFL Stats, Finishes, and Milestones',col))

#Split data into Rookies, Veterans and Young Players (Sophs)
############################################################################################################
# We will train the models on Veteran data to infer Rookie information
# Sophomore data will serve as a sanity check
#This block also separates the training data from NFL production (which is what
#we are trying to predict).

# data = pd.concat((wr_data[features],rb_data[features]))
data = wr_data

rooks = data.loc[data[('Draft', 'Draft Year')] == 2020]
vets = data.loc[data[('Draft', 'Draft Year')] <= 2016]
soph = data.loc[(data[('Draft', 'Draft Year')] > 2016) & (data[('Draft', 'Draft Year')] < 2020)]


rook_feats = rooks[features[2:-5]]
vet_feats = vets[features[2:-5]]
vet_target = vets[features[-5:]]
soph_feats = soph[features[2:-5]]
soph_target = soph[features[-5:]]



#%%Some data tidying up, replacing empty spaces with nans.
############################################################################################################
#For production and combine data we replace nans with the median value.
#This does introduce a bit of data leakage, but it's pretty safe to assume
#players would be mediocre in combine exercises they chose not to perform in.
#Also, we only have one player with missing college stats.
class fill_values:
    def empty_to_nan(series):
        temp = series.replace(np.nan,'-').replace('-',np.NaN)
        return temp
    def with_max(series):
        temp = fill_values.empty_to_nan(series)
        temp = temp.fillna(value=5+np.nanmax(temp))
        return temp
    def with_median(series):
        temp = fill_values.empty_to_nan(series)
        temp = temp.fillna(value=np.nanmedian(temp))
        return temp
    def with_zero(series):
        temp = series.replace('-',0)
        return temp

for feat in features:
    if 'BOA' in feat[1]:
        rook_feats[feat] = fill_values.with_max(rook_feats[feat])
        vet_feats[feat] = fill_values.with_max(vet_feats[feat])
        soph_feats[feat] = fill_values.with_max(soph_feats[feat])
    elif 'Combine' in feat[0] or 'Career' in feat[0]:
        rook_feats[feat] = fill_values.with_median(rook_feats[feat])
        vet_feats[feat] = fill_values.with_median(vet_feats[feat])
        soph_feats[feat] = fill_values.with_median(soph_feats[feat])
    elif 'NFL' in feat[0]:
        vet_target[feat] = vet_target[feat].replace('-',0)
        soph_target[feat] = soph_target[feat].replace('-',0)

#Dropping multilevel column names for simplicity in the next steps
vet_target.columns = vet_target.columns.droplevel()
soph_target.columns = soph_target.columns.droplevel()
vet_feats.columns = vet_feats.columns.droplevel()
soph_feats.columns = soph_feats.columns.droplevel()
rook_feats.columns = rook_feats.columns.droplevel()

# %% Function where we set labels to each player based on their
# real life fantasy performance in the NFL.
# The cut-off points are also completely arbitrary.
############################################################################################################
def set_hit(df):
    hit = np.zeros(len(df))
    for i in range(len(hit)):
        temp = df.iloc[i]
        if temp['Top24'] > 1:
            hit[i] = True
        else:
            hit[i] = False
    return hit

hit_vets = set_hit(vet_target)
hit_soph = set_hit(soph_target)

#%%Shitty plots
############################################################################################################
plot_data=vet_feats
plot_data['Hit'] = hit_vets
cols = plot_data.columns
nplots = len(cols)-2
n=1
plt.figure(figsize=(5,20))
for col in cols:
    if col != 'Hit' and col != 'Conf':
        plt.subplot(np.ceil(nplots/2),2,n)
        sns.boxplot(x=col,y='Hit',data=plot_data,orient='h')
        # plt.scatter(plot_data[col],plot_data['Hit'],c=plot_data['Hit'])
        plt.xlabel(col)
        n+=1
plt.tight_layout()




# %% Scalling data to a standard scale, onehot encoding conferences.
############################################################################################################
def encode_df(feat,encoder):
    scaler = MinMaxScaler()

    encoded = encoder.transform(feat['Conf'].values.reshape(-1,1)).toarray()
    feat = feat.drop(['Conf'],axis=1)

    feat = feat.replace('UDFA',np.nan).replace(np.nan,256)
    scaled = scaler.fit_transform(feat)

    out = np.concatenate([scaled,encoded],axis=1)

    return out 

encoder = OneHotEncoder()
encoder.fit(wr_data[('FF_Spaceman','Conf')].values.reshape(-1,1))
n_confs = encoder.categories_
X = encode_df(vet_feats,encoder)
X_rk = encode_df(rook_feats,encoder)
X_soph = encode_df(soph_feats,encoder)

#%%Define model and sampling
#Some models to test
############################################################################################################
# estimator = XGBClassifier(n_estimators=1000)
estimator = MLPClassifier()
parameters ={}
model0 = GridSearchCV(estimator, parameters,cv=10,n_jobs=-1,scoring='f1')

smote = SMOTE(sampling_strategy='not majority',k_neighbors=2,n_jobs=-1)
enn = EditedNearestNeighbours(sampling_strategy='not minority',n_neighbors=5,n_jobs=-1)
sample = SMOTEENN(sampling_strategy='auto',n_jobs=-1,smote=smote,enn=enn)
sample2 = SMOTEENN(sampling_strategy='auto',n_jobs=-1,smote=smote,enn=enn)


# split the data
X_train, X_test, y_train, y_test = train_test_split(X, hit_vets, stratify=hit_vets, test_size=0.2, random_state=42)
y_train = np.uint8(y_train)
y_test = np.uint8(y_test)
# X_train, y_train = sample.fit_resample(X_train,y_train)
# X_test, y_test = sample2.fit_resample(X_test,y_test)

sns.distplot(y_train,hist=False)
sns.distplot(y_test,hist=False)

#%%
# fit model
model = model0
model.fit(X_train, y_train)
# evaluate model
y_hat = model.predict(X_test)
conf = confusion_matrix(y_test,y_hat,normalize='pred')
tp = conf[1][1]
fn = conf[1][0]

plotConfMatrix(conf,labels=[0,1])

#%% Predictions
pred_soph_hit = model.predict(X_soph)
prod_soph_prob = model.predict_proba(X_soph)
pred_rook_hit = model.predict(X_rk)
pred_rook_prob = model.predict_proba(X_rk)

#Format results for output
############################################################################################################
soph[('','Predicted Hit')] = pred_soph_hit
rooks[('','Predicted Hit')] = pred_rook_hit
soph[('','Hit Probability')] = prod_soph_prob.T[1]
rooks[('','Hit Probability')] = pred_rook_prob.T[1]

out_feats =    [ ('Unnamed: 0_level_0','Player'),
    ('','Hit Probability'),
    ('Draft', 'DP'),
    ('Draft', 'Draft Age'),
    ('FF_Spaceman', 'Conf'),
    ('College Dominator', 'REC CD'),
    ('Breakout Age', 'WR BOA (20%)'),
    ('Career Best', 'REC/TM PA'),
    ('Career Best', 'REC Yards/TM PA'),
    ('Career Best', 'PPR/GP'),
    ('Career Best', 'SCRIM YDs/ Touch Over TM AVG')
]
for col in data['Combine'].columns[1:-1]:
    out_feats.append(('Combine',col))

out_soph = soph[out_feats]
out_soph.columns = out_soph.columns.droplevel()
out_rk = rooks[out_feats]
out_rk.columns = out_rk.columns.droplevel()

out_soph.to_excel('soph.xls')
out_rk.to_excel('rook.xls')

# %%
