import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from geopy.distance import vincenty, great_circle


def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5

def find_outliers(series):
    order = series.argsort().values
    index = (series.index)
    i = -1
    res = series < 0
    while len(series)>5 and series[index[order[i]]] > series[index[order[i-1]]]*3 and series[index[order[i-1]]] > 20:
        res[index[order[i]]] = True
        i -= 1
        print(series[index[order[i]]],series[index[order[i-1]]])        
    return res

def add_weather(train):
    wea = pd.read_csv('../input/rrv-weather-data/air_store_with_stations.csv')
    tra =  train.copy()
    tra = pd.merge(tra, wea[['station_id','prec_stations','temp_stations','air_store_id']], how='left', on = ['air_store_id'])
    sta_all = pd.DataFrame()
#    
#    for station in wea['station_id'].unique()[:]:
#    #    a = 'hyogo__kobe-kana__koube'
#        sta = pd.read_csv('../input/rrv-weather-data/1-1-16_5-31-17_Weather/'+station+'.csv')
#        sta = sta.rename(index=str, columns={'calendar_date':'visit_date'})
#        sta.visit_date = pd.to_datetime(sta.visit_date)
#        sta['station_id'] = station
#        sta_all = sta_all.append(sta[['deepest_snowfall','station_id','visit_date',]])        
#    tra = pd.merge(tra, sta_all, how='left', on=['station_id','visit_date' ])
    
    sta_all = pd.DataFrame()
    for station in wea['temp_stations'].unique()[:]:
        sta = pd.read_csv('../input/rrv-weather-data/1-1-16_5-31-17_Weather/'+station+'.csv')
        sta = sta.rename(index=str, columns={'calendar_date':'visit_date'})
        sta.visit_date = pd.to_datetime(sta.visit_date)
        sta['temp_stations'] = station
        sta_all = sta_all.append(sta[['avg_temperature','temp_stations','visit_date',]])        
    tra = pd.merge(tra, sta_all, how='left', on=['temp_stations','visit_date' ])

    sta_all = pd.DataFrame()
    for station in wea['prec_stations'].unique()[:]:
        sta = pd.read_csv('../input/rrv-weather-data/1-1-16_5-31-17_Weather/'+station+'.csv')
        sta = sta.rename(index=str, columns={'calendar_date':'visit_date'})
        sta.visit_date = pd.to_datetime(sta.visit_date)
        sta['prec_stations'] = station
        sta_all = sta_all.append(sta[['precipitation','prec_stations','visit_date',]])        
    tra = pd.merge(tra, sta_all, how='left', on=['prec_stations','visit_date' ])

    return tra


air_visit_data = pd.read_csv('../input/air_visit_data.csv')
date_info = pd.read_csv('../input/date_info.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

wkend_holidays = date_info.apply(
    (lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 3  
date_info['weight_hol'] = 1

visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)
visit_data['is_outlier'] = visit_data.groupby('air_store_id').apply(lambda g: find_outliers(g['visitors'])).values
visit_data = visit_data[visit_data['is_outlier'] == False]

wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
wmean_hol = lambda x:( (x.weight_hol * x.visitors).sum() / x.weight_hol.sum() )

visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0:'visitors'}, inplace=True) 

visitors1 = visit_data.groupby(['air_store_id', 'holiday_flg']).apply(wmean_hol).reset_index()
visitors1.rename(columns={0:'visitors'}, inplace=True)

sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')

sample_submission = sample_submission.merge(
            visitors1[visitors1.holiday_flg==1], on=('air_store_id','holiday_flg'), 
            how='left')

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), 
    how='left')['visitors_y'].values

missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), 
    on='air_store_id', how='left')['visitors_y'].values

sub2 = sample_submission[['id', 'visitors']].copy()




del air_visit_data; del date_info; del sample_submission
 
data = {
    'tra': pd.read_csv('../input/air_visit_data_with_rwm.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
#    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    'hol': pd.read_csv('../input/hol_add.csv')
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

data['ar_lgb'] = data['ar'].copy() 
data['hr_lgb'] = data['hr'].copy()
for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_dow'] = data[df]['visit_datetime'].dt.dayofweek
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df][data[df]['reserve_datetime_diff'] > 0]
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])
for df in ['ar_lgb','hr_lgb']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_dow'] = data[df]['visit_datetime'].dt.dayofweek
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df][data[df]['reserve_datetime_diff'] >7]

    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])


data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

kmeans = KMeans(n_clusters=10, random_state=0).fit(data['as'][['longitude','latitude']])
data['as']['cluster'] = kmeans.predict(data['as'][['longitude','latitude']])
data['as']['dis_cluster'] = pd.Series(map(lambda x,y,z : vincenty([x,y],kmeans.cluster_centers_[z]).km, data['as']['longitude'],data['as']['latitude'],data['as']['cluster']))

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 

stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
lbl = preprocessing.LabelEncoder()
for i in range(10):
    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 


train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])


train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)


train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']

train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']

lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
test['air_store_id2'] = lbl.transform(test['air_store_id'])

train.visit_date = pd.to_datetime(train.visit_date)
test.visit_date = pd.to_datetime(test.visit_date)



train_lgb = train.copy()
test_lgb = test.copy()
for df in ['ar','hr']:
    data[df].visit_date = pd.to_datetime(data[df].visit_date)
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])
train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2
test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2
train = add_weather(train)
#train['deepest_snowfall'] = train['deepest_snowfall'].fillna(0)
train = train.fillna(-1)
for df in ['ar_lgb','hr_lgb']:
    data[df].visit_date = pd.to_datetime(data[df].visit_date)
    train_lgb = pd.merge(train_lgb, data[df], how='left', on=['air_store_id','visit_date']) 
    test_lgb = pd.merge(test_lgb, data[df], how='left', on=['air_store_id','visit_date']) 
train_lgb['total_reserv_sum'] = train_lgb['rv1_x'] + train_lgb['rv1_y']
train_lgb['total_reserv_mean'] = (train_lgb['rv2_x'] + train_lgb['rv2_y']) / 2
train_lgb['total_reserv_dt_diff_mean'] = (train_lgb['rs2_x'] + train_lgb['rs2_y']) / 2
test_lgb['total_reserv_sum'] = test_lgb['rv1_x'] + test_lgb['rv1_y']
test_lgb['total_reserv_mean'] = (test_lgb['rv2_x'] + test_lgb['rv2_y']) / 2
test_lgb['total_reserv_dt_diff_mean'] = (test_lgb['rs2_x'] + test_lgb['rs2_y']) / 2

train_lgb = add_weather(train_lgb)
#train_lgb['deepest_snowfall'] = train_lgb['deepest_snowfall'].fillna(0)
train_lgb = train_lgb.fillna(-1)

test = add_weather(test)
test_lgb = add_weather(test_lgb)


test['rwm_1week'] = sub2['visitors'] 
test_lgb['rwm_1week'] = sub2['visitors'] 
test_lgb = test_lgb.fillna(-1)
test = test.fillna(-1)
#tmp1 = train.groupby(['air_area_name'], as_index=False)[['air_store_id']].count().rename(columns={'air_store_id':'stores_in_area'})
#train = pd.merge(train, tmp1, how='left',on=['air_area_name'])
#test = pd.merge(test, tmp1, how='left',on=['air_area_name'])

train['is_outlier'] = train.groupby('air_store_id').apply(lambda g: find_outliers(g['visitors'])).values
train = train[train['is_outlier'] == False]    
train_lgb['is_outlier'] = train_lgb.groupby('air_store_id').apply(lambda g: find_outliers(g['visitors'])).values
train_lgb = train_lgb[train_lgb['is_outlier'] == False] 
col = [c for c in train if c not in ['id' , 'air_store_id', 'visit_date','visitors',
                                           'is_outlier','station_id','temp_stations','prec_stations',]]

col1 = [c for c in col if c not in ['precipitation','avg_temperature', 'deepest_snowfall']]

col2 = [c for c in col if c not in ['rwm_1week']]
    

model2 = ensemble.RandomForestRegressor(n_estimators=200, criterion='mse',
                                            max_depth=35, min_samples_split=30, 
                                            min_samples_leaf=10, min_weight_fraction_leaf=0.0,
                                            max_features='auto', max_leaf_nodes=None, 
                                            random_state=42,
                                            min_impurity_split=None, bootstrap=True, 
                                            oob_score=False, n_jobs=-1,  
                                            verbose=0, warm_start=False)

model3 = XGBRegressor(learning_rate=0.01,  n_estimators=2000, subsample=0.8, 
                          colsample_bytree=0.8, max_depth =12
                          ,reg_alpha=0, reg_lambda=1)


model2.fit(train[col], np.log1p(train['visitors'].values))
preds2 = model2.predict(test[col])
model2.fit(train[col1], np.log1p(train['visitors'].values))
preds21 = model2.predict(test[col1])
model3.fit(train[col], np.log1p(train['visitors'].values))
preds3 = model3.predict(test[col])

del train

params = {}
params['learning_rate'] = 0.02
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['sub_feature'] = 0.8
params['num_leaves'] = 2000
params['min_data'] = 10
params['min_hessian'] = 1
params['nthread'] = 8
num_it = 350
train_c = train_lgb.copy()
test_c = test_lgb.copy() 
d_train = lgb.Dataset(train_c[col2], np.log1p(train_c['visitors']))
model = lgb.train(params, d_train, num_it)
bst = lgb.train(params, d_train, num_it)
ax = lgb.plot_importance(model, max_num_features=20)
plt.show()


preds4 = bst.predict(test_c[col2], num_iteration=num_it)

test['visitors'] = (0.5*preds2+0.5*preds21 + preds3 + preds4 )/3
test1 = test.copy()

test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
test[['id', 'visitors']].to_csv('submission_3models_v2.csv', index=False)


pro = pd.read_csv('./prophet_holidays00.csv')
test1['visitors'] = 0.8*test1['visitors'] + 0.2*np.log1p(pro['visitors'])
test1['visitors'] = np.expm1(test1['visitors']).clip(lower=0.)
test1[['id', 'visitors']].to_csv('submission_with_pro.csv', index=False)