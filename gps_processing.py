# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:21:36 2023

@author: Yutian Chen

This script is for process GPS information and generate related features.
"""

import pandas as pd
import numpy as np
import os


#------------------------------------------------------------------------------
# GPS processing functions

# Calculate Haversine distance between two sets of coordinates
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r 


# Find MRT station / primary school / shopping mall within certain distance to the HDB
def find_poi_within_distance(dt, pois, distance):
    """
    Parameters
    ----------
    dt : dataframe
        HDB dataset with GPS information.
    pois : dataframe
        MRT station / primary school / shopping mall GPS information.
    distance : float
        Find facility within this distance.    
    """
    close_pois = []
    
    for _, poi in pois.iterrows():
        
        #if abs(dt['latitude']-poi['latitude']) + abs(dt['longitude']-poi['longitude']) < 0.02:
        dist = haversine(
            dt['latitude'], dt['longitude'], poi['latitude'], poi['longitude'])
        if dist <= distance:
            close_pois.append({
                'StationName': poi['name'],
                'distance_km': dist,
                'year':poi['opening_year']
            })
    
    return close_pois


# Find the closest one to HDB
def find_min_distance(df):

    min_distances = []

    for _, row in df.iterrows():
        index = row['index']
        close_pois = row['ClosestPOIs']

        if close_pois:
            min_distance = min(poi['distance_km'] for poi in close_pois)
            min_distances.append({'index': index, 'minDistance': min_distance})
        else:
            min_distances.append({'index': index, 'minDistance': None})

    return pd.DataFrame(min_distances)


# Count the number of facilities within certain distance
def count_poi(df, distance):

    poi_counts = []

    for _, row in df.iterrows():
        index = row['index']
        close_pois = row['ClosestPOIs']

        if close_pois:
            count = sum(1 for poi in close_pois if poi['distance_km'] <= distance)
            poi_counts.append({'index': index, 'POIinDist': count})
        else:
            poi_counts.append({'index': index, 'POIinDist': 0})

    return pd.DataFrame(poi_counts)


# Count the number of MRT station within certain distance
def count_poi_for_mrt(df, distance, n):
    poi_counts = []

    for _, row in df.iterrows():
        index = row['index']
        close_pois = row['ClosestPOIs']
        year = row['year']

        if close_pois:
            count = sum(1 for poi in close_pois if poi['distance_km'] <= distance and poi['year'] <= int(year)+n) # n=0 for current MRT station, and n>0 represent in future n years
            poi_counts.append({'index': index, 'POIinDist': count})
        else:
            poi_counts.append({'index': index, 'POIinDist': 0})

    return pd.DataFrame(poi_counts)



#------------------------------------------------------------------------------
# Data import
df = pd.read_csv('train.csv')  
# df = pd.read_csv('test.csv')

df_mrtsta = pd.read_csv(os.path.join('auxiliary-data', 'sg-mrt-existing-stations.csv'))
df_mrtsta_plan = pd.read_csv(os.path.join('auxiliary-data', 'sg-mrt-planned-stations.csv'))
df_pschool = pd.read_csv(os.path.join('auxiliary-data', 'sg-primary-schools.csv'))
df_shopping = pd.read_csv(os.path.join('auxiliary-data', 'sg-shopping-malls.csv'))

df['year'] = df['rent_approval_date'].str[:4]
df = df.reset_index()



#------------------------------------------------------------------------------
# MRT station distance & MRT count within distance
df_mrtsta.drop_duplicates(inplace=True)
df_mrtsta['name'] = df_mrtsta['name'] + '_' + df_mrtsta['code']

closest_mrt = []

for _, apt in df.iterrows():
    close_pois = find_poi_within_distance(apt, df_mrtsta, 3)
    closest_mrt.append({
        'index': apt['index'],
        'ClosestPOIs': close_pois})

closest_mrt_df = pd.DataFrame(closest_mrt)
smallest_distances_df = find_min_distance(closest_mrt_df)
closest_mrt_df1 = closest_mrt_df.merge(df[['index', 'year']], on='index', how='left')
mrt_cnt = count_poi_for_mrt(closest_mrt_df1, 0.5, 0)

df_mrt = mrt_cnt.merge(smallest_distances_df, on='index', how='left')


#------------------------------------------------------------------------------
# Primary school distance
df_pschool = df_pschool.drop_duplicates()
df_pschool['opening_year'] = ''
closest_school = []

for _, apt in df.iterrows():
    close_pois = find_poi_within_distance(apt, df_pschool, 5)
    closest_school.append({
        'index': apt['index'],
        'ClosestPOIs': close_pois
    })

closest_school_df = pd.DataFrame(closest_school)
minDist_school = find_min_distance(closest_school_df)
#school_cnt = count_poi(closest_school_df, 1)


#------------------------------------------------------------------------------
# Shopping mall distance
df_shopping = df_shopping.drop_duplicates()
df_shopping['opening_year'] = ''
closest_mall = []

for _, apt in df.iterrows():
    close_pois = find_poi_within_distance(apt, df_shopping, 10)
    closest_mall.append({
        'index': apt['index'],
        'ClosestPOIs': close_pois
    })

closest_mall_df = pd.DataFrame(closest_mall)
minDist_mall = find_min_distance(closest_mall_df)


#------------------------------------------------------------------------------
# Merge results
df_result = df_mrt.merge(minDist_school, on='index', how='left')
df_result = df_result.merge(minDist_mall, on='index', how='left')
df_result.columns = ['index', 'MRT_cnt_500', 'minDist_MRT', 'minDist_school', 'minDist_mall']

df_result.to_csv(os.path.join('auxiliary-data','GPS_processed.csv'), index=False)








