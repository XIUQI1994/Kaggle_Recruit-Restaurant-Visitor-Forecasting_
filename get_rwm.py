
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:13:18 2018

@author: xiuqi
"""

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import datetime,date,timedelta
from sklearn.metrics import make_scorer, accuracy_score,mean_squared_error,mean_absolute_error,r2_score


def find_outliers(series):
    series = series.map(pd.np.expm1)
    order = series.argsort().values
    index = (series.index)
    i = -1
    res = series < 0
    while len(series)>5 and series[index[order[i]]] > series[index[order[i-1]]]*3 and series[index[order[i-1]]] > 20:
        print(series[index[order[i]]],series[index[order[i-1]]])
        res[index[order[i]]] = True
        i -= 1        
    return res

    
air_visit_data = pd.read_csv('../input/air_visit_data.csv')
air_store_info =  pd.read_csv('../input/air_store_info.csv')
date_info = pd.read_csv('../input/date_info.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

wkend_holidays = date_info.apply(
    (lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0


air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])
date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date'])
date_info = date_info.rename(columns={'calendar_date':'visit_date'})
air_visit_data['visitors'] = air_visit_data.visitors.map(pd.np.log1p)

datelist = pd.Series(pd.date_range(pd.datetime(2016,2,1), pd.datetime(2017,4,15))).dt.date[::-1][::7][::-1]

dt1 = timedelta(days=7)
dt2 = timedelta(days=39)

train = air_visit_data.copy()

error = [[],[]]
rwm1 = pd.DataFrame()
rwm2 = pd.DataFrame()
for date in datelist[:]:    
    train_c = air_visit_data[air_visit_data['visit_date'] <= date]
    train_c['is_outlier'] = train_c.groupby('air_store_id').apply(lambda g: find_outliers(g['visitors'])).values
    train_c = train_c[train_c['is_outlier'] == False]    

    val_c = air_visit_data[air_visit_data['visit_date'] <= date+dt1]
    val_c = val_c[val_c['visit_date'] >date]
    test_c = air_visit_data[air_visit_data['visit_date'] <=date+dt2]
    test_c = test_c[test_c['visit_date'] > date+dt1]
    date_info_c = date_info[date_info['visit_date'] <= date]
    date_info_c['weight'] = ((date_info_c.index + 1) / len(date_info_c))**3
    date_info_c['weight_hol'] = 1
    visit_data = train_c.merge(date_info_c, on='visit_date', how='left')

    wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
    wmean_hol = lambda x:( (x.weight_hol * x.visitors).sum() / x.weight_hol.sum() )
    wmean_hol = lambda x:( x.visitors.median() )
    visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
    visitors.rename(columns={0:'visitors'}, inplace=True) # cumbersome, should be better ways.

    visitors1 = visit_data.groupby(['air_store_id', 'holiday_flg']).apply(wmean_hol).reset_index()
    visitors1.rename(columns={0:'visitors'}, inplace=True) # cumbersome, should be better ways.    

    val_c = val_c.merge(date_info, on='visit_date', how='left')
    test_c = test_c.merge(date_info, on='visit_date', how='left')
    
    i = 0
    for df in [ val_c, test_c]:
        real = df['visitors'].values    
        df.drop('visitors', axis=1, inplace=True)
        

        df = df.merge(
            visitors1[visitors1.holiday_flg==1], on=('air_store_id','holiday_flg'), 
            how='left')
        
        missings = df.visitors.isnull()
        df.loc[missings, 'visitors'] = df[missings].merge(
            visitors[visitors.holiday_flg==0], on=('air_store_id', 'day_of_week'), 
            how='left')['visitors_y'].values
        
        missings = df.visitors.isnull()
        df.loc[missings, 'visitors'] = df[missings].merge(
            visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), 
            on='air_store_id', how='left')['visitors_y'].values
        
        missings = df.visitors.isnull()
        df.loc[missings, 'visitors'] = df[missings].merge(
            visitors[['day_of_week','holiday_flg', 'visitors']].groupby(['day_of_week','holiday_flg']).mean().reset_index(), 
            on=['day_of_week','holiday_flg'], how='left')['visitors_y'].values
                
        i += 1
        if i == 1:
            rwm1 = rwm1.append(df[['air_store_id', 'visitors','visit_date']])
        else:
            rwm2 = rwm2.append(df[['air_store_id', 'visitors','visit_date']])            



rwm1.rename(columns={'visitors':'rwm_1week'}, inplace=True)
rwm2.rename(columns={'visitors':'rwm_39days'}, inplace=True)
train = train.merge(rwm1, on = ('air_store_id','visit_date'), how = 'left')
train['visitors'] =train.visitors.map(pd.np.expm1)
train.to_csv('../input/air_visit_data_with_rwm.csv', index=False)

#train = train.merge(rwm2, on = ('air_store_id','visit_date'), how = 'left')
#train['visitors'] =train.visitors.map(pd.np.expm1)
#train.to_csv('../input/air_visit_data_with_rwm39days.csv', index=False)
