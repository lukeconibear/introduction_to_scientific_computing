# OpenAQ

Download OpenAQ data and convert to Pandas DataFrame

## 1. Download the data

Using `download.py` as below, data in `.ndjson.gz` format for a given date range (e.g. 1st Jan 2015 to 31st Dec 2015) using:

```
python download.py 2015-01-01 2015-12-31
```  

#!/usr/bin/env python3
"""
Download script for all OpenAQ data
Credit to https://github.com/barronh/scrapenaq
Downloads all (.ndjson.gz) files between dates
Example:
    python download.py 2020-01-01 2020-12-31
"""

import pandas as pd
import urllib.request
import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('startdate', help='Find ndjson.gz created >= startdate')
parser.add_argument('enddate', help='Find ndjson.gz created <= enddate')

args = parser.parse_args()

dates = pd.date_range(args.startdate, args.enddate)
keyre = re.compile('<Key>(.+?)</Key>')
BROOT = 'openaq-fetches.s3.amazonaws.com/'

for date in dates:
    xrpath = (
        'https://{}?delimiter=%2F&'.format(BROOT) +
        'prefix=realtime-gzipped%2F{}%2F'.format(date.strftime('%F'))
    )
    xmlpath = BROOT + date.strftime('realtime-gzipped/%F.xml')
    zippedpath = BROOT + date.strftime('realtime-gzipped/%F')
    
    os.makedirs(zippedpath, exist_ok=True)

    if os.path.exists(xmlpath):
        print('Keeping cached', xmlpath)
    else:
        urllib.request.urlretrieve(xrpath, xmlpath)

    xmltxt = open(xmlpath, mode='r').read()
    keys = keyre.findall(xmltxt)
    
    for key in keys:
        url = 'https://' + BROOT + key
        outpath = BROOT + key
        if os.path.exists(outpath):
            print('Keeping cached', outpath)
        else:
            urllib.request.urlretrieve(url, outpath)

## 2. Convert to a Pandas DataFrame
Using `convert.py` below, convert the `.ndjson.gz` data to a Pandas DataFrame for a given year (e.g. 2015) using:  
```
python convert.py 2015
```

#!/usr/bin/env python3
"""
Convert the default format for OpenAQ data (.ndjson.gz) to Pandas DataFrame for a given year
"""

import glob
import gzip
import ndjson
from pandas.io.json import json_normalize
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('year')
args = parser.parse_args()

year = args.year

path = '~' # change to this where the data is
files = sorted(glob.glob('openaq-fetches.s3.amazonaws.com/realtime-gzipped/' + year + '*/*'))

df_list = []
for file in files:
    with gzip.open(file, 'rb') as ds:
        data_ndjson = ds.read()

    data_json = ndjson.loads(data_ndjson)

    df_list.append(json_normalize(data_json))
    
df = pd.concat(df_list, sort=False)

df.set_index('date.utc', inplace=True)
df.index = pd.to_datetime(df.index)

df.to_csv(path + 'openaq_data_' + year + '.csv')

## To analyse the OpenAQ data

### Option 1: eagerly load into memory using Pandas

import pandas as pd

df = pd.read_csv(
    '/nfs/b0004/Users/earlacoa/openaq/shared/openaq_data_2013-2020_india.csv', 
    parse_dates=['date.utc'],
    usecols=['date.utc', 'parameter', 'value', 'unit', 'coordinates.latitude', 'coordinates.longitude', 'city', 'country'],
    index_col='date.utc'
)

df['2020-02-01':'2020-04-30']

### Option 2: lazily load using Dask with parquet files

import dask.dataframe as dd
from dask.distributed import Client
client = Client()
client

df = dd.read_parquet(
    '/nfs/b0004/Users/earlacoa/openaq/shared/openaq_data_2013-2020_india.parquet',
    columns=['parameter', 'value', 'unit', 'coordinates.latitude', 'coordinates.longitude', 'city', 'country']
)

To compute tasks use the `.compute()` method

client.close()

## Convert a dataset to the OpenAQ format

import glob
import pandas as pd
import xarray as xr
import numpy as np

df_openaq = pd.read_csv(
    '/nfs/b0122/Users/earlacoa/openaq/csv/openaq_data_2015_noduplicates.csv',
    index_col="date.utc",
    parse_dates=True
)

df_obs_summaries = {'2014': [], '2015': [], '2016': [], '2017': [], '2018': [], '2019': [], '2020': []}
china_obs_files = glob.glob('/nfs/a68/earlacoa/china_measurements_corrected/*nc')
parameters = {'CO': 'co', 'NO2': 'no2', 'O3': 'o3', 'PM10': 'pm10', 'PM2.5': 'pm25', 'SO2': 'so2'}
years = ['2014', '2015', '2016', '2017', '2018', '2019', '2020']

for china_obs_file in china_obs_files:
    ds_obs = xr.open_dataset(china_obs_file)
    
    for parameter in parameters.keys():
        dict_obs = {
            'date.utc': ds_obs.time.values,
            'city': ds_obs.city,
            'unit': 'µg/m³',
            'value': ds_obs[parameter].values,
            'country': 'CN',
            'location': ds_obs.name,
            'parameter': parameters[parameter],
            'sourceName': 'China measurements',
            'sourceType': 'government',
            'date.local': ds_obs.time.values + np.timedelta64(8, 'h'),
            'coordinates.latitude': ds_obs.lat,
            'coordinates.longitude': ds_obs.lon, 
            'averagingPeriod.unit': 'hours',
            'averagingPeriod.value': 1
        }
        df_obs = pd.DataFrame.from_dict(dict_obs)
        df_obs.set_index('date.utc', inplace=True)
        
        for year in years:
            df_obs_summaries[year].append(df_obs[year])
        
    ds_obs.close()

for year in years:
    df_obs_summaries_concat = pd.concat(df_obs_summaries[year])
    df_obs_summaries_concat.to_csv(f'/nfs/a68/earlacoa/china_measurements_corrected/df_obs_summary_{year}.csv')