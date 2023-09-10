import numpy as np
import modin.pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()

from tensorflow import keras
from tensorflow.keras import layers
from io import StringIO
from IPython.display import Image, display
import time
from sklearnx.metrics import mean_squared_error as mse
from pandas.core.dtypes.common import classes
import plotly.offline as offline
from sklearnx.preprocessing import MinMaxScaler
import plotly
import plotly.figure_factory as ff
from urllib.request import urlopen
import json
import plotly.express as px

import ray
ray.init()


population = pd.read_excel(file_path)
metrics = pd.read_csv(file_path)
predicted = metrics = pd.read_csv(file_path)
actual = pd.read_csv(file_path)

cols = predicted.columns
cols = cols[:-3]
date = predicted['date']
predicted.drop(['date'],axis = 1,inplace = True)
predicted.dropna(how = 'all',axis = 0,inplace = True)
predicted = pd.merge(predicted,date,left_index=True, right_index=True)
predicted.reset_index(inplace = True,drop = True)
df = pd.DataFrame(columns = ['Date','Values','FIPS'])

for i in cols:
  temp = predicted[[i,'date']]
  temp.rename(columns = {str(i) : 'Values','date' : 'Date'},inplace = True)
  temp["FIPS"] = i
  df = pd.concat([df,temp])
  df.reset_index(inplace = True,drop = True)
df.dropna(subset = 'Values',inplace = True)
df.reset_index(inplace = True,drop = True)
df['FIPS'] = df['FIPS'].astype('float')
df['FIPS'] = df['FIPS'].astype('int')
df = df.merge(population[['2015 POPULATION','FIPS CODE']],left_on = 'FIPS',right_on = 'FIPS CODE',how = 'inner')
df.drop('FIPS CODE',inplace = True,axis = 1)
df['Cases/Population'] = df['Values']/df['2015 POPULATION']
df.dropna(subset = ['Cases/Population'],inplace = True)
df.reset_index(inplace = True,drop = True)

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
scaler = MinMaxScaler((0,100))
#df =pd.read_csv('https://raw.githubusercontent.com/SparshRastogi/Covid-19-Risk-Calculator/main/2021%20Month1%20Risk.csv')#,dtype={"fips": str})
#df = df[0]

df = df[df['Date'] == '2022-01-01']
#for i in range(1,10):
# df =  pd.read_csv('/content/drive/MyDrive/Intel OneAPI/Predictions.csv')

df['FIPS']=df['FIPS'].apply(lambda x: '{0:0>5}'.format(x) )
df['Cases/Population'] = scaler.fit_transform(np.array(df['Cases/Population']).reshape(-1,1))
fig = px.choropleth(df,geojson = counties,locations='FIPS', color='Cases/Population',
                           color_continuous_scale= ['green','light blue','orange','purple','red'],
                          range_color=(df['Cases/Population'].min(),df['Cases/Population'].max()),
                           scope="usa")

fig.update_layout(margin=dict(l=20,r=0,b=0,t=70,pad=0),paper_bgcolor="white",height= 700,title_text = 'Supervision of daily Covid-19 cases constituency wise',font_size=18)
fig.show()
