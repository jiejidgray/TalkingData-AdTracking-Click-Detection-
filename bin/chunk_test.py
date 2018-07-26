import pandas as pd
import numpy as np
import gc

import joblib
import xgboost as xgb
import matplotlib
matplotlib.use('TkAgg')
from xgboost import plot_importance
from matplotlib import pyplot as plt
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

def categ(dataset):
    variables = ['ip', 'app', 'device', 'os', 'channel']
    for v in variables:
        dataset[v] = dataset[v].astype('category')
    return dataset


def featuring(df):
    gp = df[['ip','app','channel']].groupby(by=['ip','app'])['channel'].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    df = df.merge(gp, on=['ip','app'],how='left')
    del gp

    gc.collect()
    gp = df[['ip','app','device','channel']].groupby(by=['ip','app','device'])['channel'].count().reset_index().rename(index=str, columns={'channel':'device_count'})
    df = df.merge(gp,on=['ip','app','device'],how='left')
    del gp
    gc.collect()
    gp = df[['ip','day','channel','hour']].groupby(by=['ip','day','channel'])['hour'].var().reset_index().rename(index=str, columns={'hour':'hour_mean'})
    df = df.merge(gp,on=['ip','day','channel'],how='left')
    del gp
    gc.collect()
    print(df.head())
    return df

def train_model(train_df,debug,idx_chunk,ratio):
    global thisM
    params = {
    'eta': 0.01,
    'max_depth':5,
    'subsumple':0.7,
    #'objective':'binary:logistic',
    'objective':'reg:linear',
    'eval_metric':'auc',
    'scale_pos_weight':ratio,
    'nthread':8,
    'silent':1
    }
    #-----Training
    y = train_df['is_attributed']
    train_df.drop(['is_attributed'], axis = 1, inplace = True)
    gc.collect()
    tracking_predictors = ['ip','app','os','device','channel','ip_app_count','device_count','hour_mean']
    train = pd.DataFrame(train_df[tracking_predictors])
    del train_df
    gc.collect()
    train.info(memory_usage='deep')
    if debug:
        train_x, test_x, train_y,test_y = train_test_split(train,y,test_size=0.3,random_state=99)
        xgb_train = xgb.DMatrix(train_x,train_y)
        del train_x,test_x,train_y,test_y,train,y
        gc.collect()
        watchlist = [(xgb_train, 'train')]
        verbose = 1
        earlystop = 2
        nrounds = 14
        #earlystop = 20
        #nrounds = 20
        #verbose = 2
    else:
        xgb_train = xgb.DMatrix(train, y)
        del train, y
        gc.collect()
        watchlist = [(xgb_train, 'train')]
        verbose = 1
        earlystop = None
        nrounds = 14

    if idx_chunk == 0:
        model = xgb.train(params,xgb_train,nrounds,watchlist,maximize=True, early_stopping_rounds = 10, verbose_eval=5)
        thisM = model
        joblib.dump(model, "pima.joblib.dat")
    else:
        model = xgb.train(params,xgb_train,nrounds,watchlist,maximize=True, early_stopping_rounds = 10, verbose_eval=5,xgb_model=thisM)
        thisM = model
        joblib.dump(model, "pima.joblib.dat")

    return model

def prediction(test_df,model):
    sub = pd.DataFrame()
    sub['click_id']=test_df['click_id'].astype('int')
    test_df.drop(['click_id'], axis=1, inplace=True)
    gc.collect()
    print(test_df.head())
    tracking_predictors = ['ip','app','os','device','channel','ip_app_count','device_count','hour_mean']
    test = pd.DataFrame(test_df[tracking_predictors])
    xgb_test = xgb.DMatrix(test)
    sub['is_attributed']= model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
    print sum(sub['is_attributed']<0)
    sub.to_csv('val%d.csv'%(fileno),index=False,float_format='%.9f')
    plot_importance(model)
    plt.show()

def main():
        dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }

        #df_attributed= pd.DataFrame()
        train_data = pd.DataFrame()
        chunksize = 18490389
        print('loading train data...')
        #train_data = pd.read_csv('/Users/jijie/isi5/webeye/project/bin/train_sample.csv', dtype = dtypes, usecols=['ip','app','device','os','channel','click_time','is_attributed'],header = 0)



        train_data = pd.read_csv('/Users/jijie/isi5/webeye/project/kaggle-talkingdata2/train.csv', chunksize=chunksize, dtype=dtypes,usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
        df_attributed=pd.DataFrame()
        for idx_chunk, chunk in enumerate(train_data):
            filtered = (chunk[(np.where(chunk['is_attributed']==1, True, False))])
            df_attributed = pd.concat([df_attributed, filtered], ignore_index=True)
            ratio = (len(chunk)-len(df_attributed))/len(df_attributed)
            print ratio
            print len(df_attributed)
            print len(chunk)
            print('chunk {} is being processing...'.format(idx_chunk))
            #chunk.info(memory_usage='deep')
            chunk = timefeature(chunk)
            #chunk = categ(chunk)
            chunk.info(memory_usage='deep')
            chunk = featuring(chunk)


            print('model{} is training now...'.format(idx_chunk))
            myModel = train_model(chunk,debug,idx_chunk,ratio)

            del chunk
            gc.collect()
            #df_attributed.info()

def predic_test():
        dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }
        loaded_model = joblib.load("pima.joblib.dat")
        print('model done! loading test data...')
        if debug:
            test_data = pd.read_csv('/Users/jijie/isi5/webeye/project/bin/test.csv', dtype=dtypes,nrows= 100000,header = 0,usecols=['ip','app','device','os', 'channel', 'click_time','click_id'])
        else:
            test_data = pd.read_csv('/Users/jijie/isi5/webeye/project/bin/test.csv', dtype=dtypes,header = 0,usecols=['ip','app','device','os', 'channel', 'click_time','click_id'])


        test_data.info(memory_usage='deep')
        print('extract test data...')
        test_data = timefeature(test_data)
        #categ(test_data)
        test_df = featuring(test_data)

        gc.collect()
        prediction(test_df,loaded_model)

filono = 1
debug = False
main()  # training model
predic_test() # predict full test dataset if debug =True
