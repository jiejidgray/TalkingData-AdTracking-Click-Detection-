import gc
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

fileno = 7

import xgboost as xgb
import matplotlib
matplotlib.use('TkAgg')
from xgboost import plot_importance
from matplotlib import pyplot as plt
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def timefeature(df):
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['minute'] = pd.to_datetime(df.click_time).dt.minute.astype('uint8')
    df['second'] = pd.to_datetime(df.click_time).dt.second.astype('uint8')
    df.drop(['click_time'], axis = 1, inplace = True)
    print('after extracted, df head')
    print(df.head())
    return df


#def CrossValidationScore:



def SaveFile(submitID, testSubmit, fileName="submit.csv"):
    content="click_id,is_attributed";
    for i in range(submitID.shape[0]):
        content+="\n"+str(submitID[i])+","+str(testSubmit[i])
    file=open(fileName,"w")
    file.write(content)
    file.close()


#my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())


train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time','click_id']

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }



#============== load the training data ===============
train_data = pd.read_csv('/Users/jijie/isi5/webeye/project/bin/train_sample.csv', dtype = dtypes, header = 0)
train_data.info(memory_usage='deep')

nrow_train = len(train_data)
# show the head of train data
print('train data head')
print(train_data.head())
#============== load the test data=================
test_data = pd.read_csv('/Users/jijie/isi5/webeye/project/bin/test.csv', nrows= 100000,header = 0)
test_data.info(memory_usage='deep')
# show the head of test data
print('test data head')
print(test_data.head())

#================== set the categorical variables ==================

variables = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    train_data[v] = train_data[v].astype('category')
    test_data[v]=test_data[v].astype('category')

train_data['is_attributed']=train_data['is_attributed'].astype('category')


#================= extract time ==================
timefeature(train_data)
timefeature(test_data)
gc.collect()

# print train_data['is_attributed'].describe()


#================ data cleaning =================
somme = train_data['attributed_time'].isnull().sum()
print somme
train_data.drop(['attributed_time'], axis = 1, inplace = True)
gc.collect()

#confidence rate   is_attributed/ precondition

# frequence days

y = train_data['is_attributed']
train_data.drop(['is_attributed'], axis = 1, inplace = True)
gc.collect()


#
#
sub = pd.DataFrame()
sub['click_id']=test_data['click_id'].astype('int')
test_data.drop(['click_id'], axis=1, inplace=True)
gc.collect()
test_data.info(memory_usage='deep')
train_data = train_data.append(test_data)
del test_data
gc.collect()

#================= add features (feature engineering) ====================
#------ clicks by ip
gp = train_data[['ip','app','channel']].groupby(by=['ip','app'])['channel'].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_data = train_data.merge(gp, on=['ip','app'],how='left')
del gp
gc.collect()
gp = train_data[['ip','app','device','channel']].groupby(by=['ip','app','device'])['channel'].count().reset_index().rename(index=str, columns={'channel':'device_count'})
train_data = train_data.merge(gp,on=['ip','app','device'],how='left')
del gp
gc.collect()
gp = train_data[['ip','day','channel','hour']].groupby(by=['ip','day','channel'])['hour'].var().reset_index().rename(index=str, columns={'hour':'hour_mean'})
train_data = train_data.merge(gp,on=['ip','day','channel'],how='left')
del gp
gc.collect()

# feature with different predictors


train = train_data[:nrow_train]
val = train_data[(nrow_train-10000):nrow_train]
test = train_data[nrow_train:]
del train_data
gc.collect()

print("train size: ", len(train))
print("valid size: ", len(val))
print("test size : ", len(test))


tracking_predictors = ['ip','app','os','device','channel','ip_app_count','device_count','hour_mean']
train = pd.DataFrame(train[tracking_predictors])

test = pd.DataFrame(test[tracking_predictors])

#X_train,X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.4, random_state=0)


#========= define model==============
#tracking_model = LogisticRegression()
params = {
'eta': 0.01,
'max_depth':5,
'subsumple':0.7,
#'objective':'binary:logistic',
'objective':'reg:linear',
'eval_metric':'auc',
'nthread':8,
'silent':1
}
#-----Training
train_x, test_x, train_y , test_y = train_test_split(train,y,test_size=0.3,random_state=99)
xgb_train = xgb.DMatrix(train_x,train_y)
xgb_test = xgb.DMatrix(test)
del train_x,test_x,train_y,test_y
gc.collect()
watchlist = [(xgb_train,'train')]
model = xgb.train(params,xgb_train,200,watchlist,maximize=True, early_stopping_rounds = 25, verbose_eval=5)

#========= fit model================
#tracking_model.fit(X,train_y)

sub['is_attributed']= model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
print sum(sub['is_attributed']<0)

#=========== submission file ============
sub.to_csv('val%d.csv'%(fileno),index=False,float_format='%.9f')

plot_importance(model)
plt.show()
#print val_predict.ptp(axis=None, out=None)

#========= cross validation===================
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(my_pipeline, X, train_y, scoring='neg_mean_absolute_error',cv= 5)
#print(scores)
#print('Mean Absolute Error %2f' %(-1 * scores.mean()))
