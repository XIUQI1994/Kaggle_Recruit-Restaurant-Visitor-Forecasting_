#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 14:40:49 2018
using prophet to predict
@author: xiuqi
"""

from fbprophet import Prophet
import numpy as np
import pandas as pd
from sklearn import ensemble, neighbors, linear_model, metrics, preprocessing
from datetime import datetime
from matplotlib import pyplot as plt


def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5


data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
#data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
sub = data['tes'].copy()
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

train = data['tra'].copy()
stores_all = data['tes']['air_store_id'].unique()
for i, store_id in enumerate(stores_all[:]):
    store1 = train[train['air_store_id'] == store_id]
    store1['visitors'] = store1.visitors.map(pd.np.log1p)
    store1.drop(['air_store_id'], axis=1, inplace=True)
    store1 = store1.rename(index=str, columns={'visit_date':'ds','visitors':'y'})

    
    holidays = pd.DataFrame({'holiday':'holiday',
                             'ds':data['hol']['visit_date'][data['hol']['holiday_flg']==1],
                             'lower_window':0,'upper_window':0.})
#decrease scale if don't have much data    
    if len(store1) < 200:
        m = Prophet(holidays=holidays, changepoint_prior_scale= 0.005)
    else:    
        m = Prophet(holidays=holidays)

    m.fit(store1)
    
    future = m.make_future_dataframe(periods=90)
    future.tail()
    forecast = m.predict(future)
    m.plot(forecast)
    m.plot_components(forecast)
    forecast = forecast.set_index('ds')
    forecast['yhat'] = forecast.yhat.map(pd.np.expm1)
    sub.loc[39*i:39*(i+1)-1, 'visitors'] = forecast['20170423':'20170531']['yhat'].values
    
sub['visitors'] = sub['visitors'].clip(lower=1.)
sub[['id', 'visitors']].to_csv('prophet_holidays00_scale.csv', index=False)    
