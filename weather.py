#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:25:07 2018

@author: xiuqi
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from dateutil.parser import parse as date_parse
from collections import Counter
from pprint import pprint as pp
from tqdm import tqdm, tqdm_notebook
from IPython.display import display
import warnings
import os
import sys
from geopy.distance import vincenty, great_circle

###############################################
# Import Plotting Assets
###############################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.pyplot import subplot, figure
import seaborn as sns
import folium
from folium import plugins as folium_plugins
from folium import features

###########

plt.interactive(False)
sns.set_style('whitegrid')
pd.set_option('display.expand_frame_repr', False)
warnings.simplefilter(action='ignore', category=FutureWarning)

weather_set_dir = '../input/rrv-weather-data'
weather_data_dir = '{}/1-1-16_5-31-17_Weather/1-1-16_5-31-17_Weather'.format(weather_set_dir)
original_data_dir = '../input/'

weather_columns = [
    'avg_temperature', 'high_temperature', 'low_temperature', 'precipitation',
    'hours_sunlight', 'solar_radiation', 'deepest_snowfall', 'total_snowfall', 'avg_wind_speed',
    'avg_vapor_pressure', 'avg_local_pressure', 'avg_humidity', 'avg_sea_pressure', 
    'cloud_cover'
]

air_store_info = pd.read_csv('{}/air_store_info_with_nearest_active_station.csv'.format(weather_set_dir))
hpg_store_info = pd.read_csv('{}/hpg_store_info_with_nearest_active_station.csv'.format(weather_set_dir))


air_station_distances = pd.read_csv('{}/air_station_distances.csv'.format(weather_set_dir))
hpg_station_distances = pd.read_csv('{}/hpg_station_distances.csv'.format(weather_set_dir))

weather_stations = pd.read_csv('{}/weather_stations.csv'.format(weather_set_dir))
nearby_active_stations = pd.read_csv('{}/nearby_active_stations.csv'.format(weather_set_dir))
feature_manifest = pd.read_csv('{}/feature_manifest.csv'.format(weather_set_dir))

original_as_info = pd.read_csv('{}/air_store_info.csv'.format(original_data_dir))
original_hs_info = pd.read_csv('{}/hpg_store_info.csv'.format(original_data_dir))

display(original_as_info.head(5))
display(original_hs_info.head(5))

print('Air Equal: {}'.format(original_as_info.equals(air_store_info[original_as_info.columns])))
print('HPG Equal: {}'.format(original_hs_info.equals(hpg_store_info[original_hs_info.columns])))

store = hpg_store_info.iloc[0]
lat_str, lon_str = store['latitude_str'], store['longitude_str']
lookup_coords = '({}, {})'.format(lat_str, lon_str).replace('"', '')
print(lookup_coords)

distances = hpg_station_distances[lookup_coords]
print(distances.values[:5])

closest_station_distance = distances.min()
print('Distance to Closest Station: {} km'.format(closest_station_distance))

ids = hpg_station_distances['station_id'].values
closest_station_id = ids[distances.tolist().index(closest_station_distance)]
print('Closest Station ID: {}'.format(closest_station_id))

stations_in_range = [(ids[_], distances[_]) for _ in range(len(distances)) if distances[_] <= 17]
stations_in_range = sorted(stations_in_range, key=lambda _: _[1], reverse=False)
pp(stations_in_range)

def select_stations(latitude_str, longitude_str, distance_df, effective_range=17.0, date_floor=None, top_n=None):
    if longitude_str is not None:
        _lookup_coords = '({}, {})'.format(latitude_str, longitude_str).replace('"', '')
    else:
        _lookup_coords = latitude_str
        
    _ids, _distances = distance_df['station_id'].values, distance_df[_lookup_coords]
    _result = [(_ids[_], _distances[_]) for _ in range(len(_ids)) if _distances[_] <= effective_range]
    
    if date_floor is not None and isinstance(date_floor, datetime):
        _result = [_ for _ in _result if '____' not in _[0] or date_parse(_[0].split('____')[1]) > date_floor]

    return sorted(_result, key=lambda _: _[1])[:top_n]

_test_0 = select_stations(lat_str, lon_str, hpg_station_distances)
_test_1 = select_stations(lat_str, lon_str, hpg_station_distances, date_floor=date_parse('2017-5-31'))
_test_2 = select_stations(lat_str, lon_str, hpg_station_distances, date_floor=date_parse('2017-5-31'), top_n=2)
_test_3 = select_stations(lat_str, lon_str, hpg_station_distances, date_floor=date_parse('1975-12-13'))

assert(_test_0 == stations_in_range)
assert(_test_1 == stations_in_range[:4] + [stations_in_range[-1]])
assert(_test_2 == stations_in_range[:2])
assert(_test_3 == stations_in_range) 

all_ranges, d_floor = ['10', '17', '20'], date_parse('2017-5-31')
separate_results, combined_results = {_: [] for _ in all_ranges}, {_: [] for _ in all_ranges}

for sys_distances in [air_station_distances, hpg_station_distances]:
    for coords in tqdm(sys_distances.columns.values[3:], leave=False):
        for e_range in all_ranges:
            res = select_stations(coords, None, sys_distances, effective_range=int(e_range), date_floor=d_floor)            
            separate_results[e_range].extend([_[0] for _ in res if _[0] not in separate_results[e_range]])
            combined_results[e_range].append([_[0] for _ in res])

print('#' * 30 + ' separate_results ' + '#' * 30)
for _r in all_ranges:
    print('{}     {}'.format(_r, len(separate_results[str(_r)])))
print('#' * 30 + ' combined_results ' + '#' * 30)
for _r in all_ranges:
    print('{}     {}'.format(_r, len(combined_results[str(_r)]))) 


#air_a271c9ba19e81d17  484
air_store_info['temp_stations'] = air_store_info['station_id']  
store = air_store_info.iloc[484]
lat_str, lon_str = store['latitude_str'], store['longitude_str']
lookup_coords = '({}, {})'.format(lat_str, lon_str).replace('"', '')
coords_air =  air_station_distances.columns.values[3:]
index = list(coords_air).index(lookup_coords)
for station in combined_results['20'][index]:
    loc = feature_manifest.loc[feature_manifest['id'].isin([station])].index
    if feature_manifest.loc[loc, 'avg_temperature'].values >= 1:
        air_store_info.loc[index, 'temp_stations'] = station
        break

#
n = 0
air_store_info['temp_stations'] = air_store_info['station_id']  
for index1, store in air_store_info.iterrows():  
    lat_str, lon_str = store['latitude_str'], store['longitude_str']
    lookup_coords = '({}, {})'.format(lat_str, lon_str).replace('"', '')
    coords_air =  air_station_distances.columns.values[3:]
    index = list(coords_air).index(lookup_coords)
    for station in combined_results['20'][index]:
        loc = feature_manifest.loc[feature_manifest['id'].isin([station])].index
        if feature_manifest.loc[loc, 'avg_temperature'].values >= 0.98:
            air_store_info.loc[index1, 'temp_stations'] = station
            n += 1
            break

print(n)

n = 0        
air_store_info['prec_stations'] = air_store_info['station_id']  
for index1, store in air_store_info.iterrows():  
    lat_str, lon_str = store['latitude_str'], store['longitude_str']
    lookup_coords = '({}, {})'.format(lat_str, lon_str).replace('"', '')
    coords_air =  air_station_distances.columns.values[3:]
    index = list(coords_air).index(lookup_coords)
    for station in combined_results['20'][index]:
        loc = feature_manifest.loc[feature_manifest['id'].isin([station])].index
        if feature_manifest.loc[loc, 'precipitation'].values > 0.95:
            air_store_info.loc[index1, 'prec_stations'] = station
            n += 1
            break
print(n)
air_store_info.to_csv('air_store_with_stations.csv', index=False)              
