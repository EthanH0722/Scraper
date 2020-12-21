# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:15:51 2020

@author: hycwy
"""
import csv
import pandas as pd
import urllib3
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=TSLA&interval=60min&slice=year2month12&adjusted=false&apikey=XG7ZQD2HHTFX4WEV'
http = urllib3.PoolManager()
response = http.request('GET', url)
print(response.status)

#print(response.data.decode())

Stockdata = response.data.decode()

datain = Stockdata.split('\r\n')

writeindata = datain[1:-1]
#%%
timel=[]
openl=[]
highl=[]
lowl=[]
closel=[]
volumel=[]
#%%
for i in writeindata:
    onelist = i.split(',')
    time_ = onelist[0]
    open_ = onelist[1]
    high_ = onelist[2]
    low_ = onelist[3]
    close_ = onelist[4]
    volume_ = onelist[5]
    timel.append(time_)
    openl.append(open_)
    highl.append(high_)
    lowl.append(low_)
    closel.append(close_)
    volumel.append(volume_)
print(len(volumel))
#%%
df = pd.DataFrame({'time':timel,'open':openl,'high':highl,'low':lowl,'close':closel,'volume':volumel})

df.to_csv('tesla2year.csv',index = False, sep=',')
#%%
