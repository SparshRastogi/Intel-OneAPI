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


import ray
ray.init()

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-SEQUENCE_SIZE):
        #print(i)
        window = obs[i:(i+SEQUENCE_SIZE)]
        after_window = obs[i+SEQUENCE_SIZE]
        window = [[x] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)

    return np.array(x),np.array(y)

model_path = 'model_path'
model = keras.models.load_model(model_path)


df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')
df['date'] = pd.to_datetime(df['date'])
counties = list(df['fips'].unique())
counties.sort()
#mobility_la['date'] = pd.to_datetime(mobility_la['date'])
metrics = pd.DataFrame(columns = ['Fips','RMSE','Mean Cases','Relative Error','Number of Instances','Minimum','Median','10%','25%','75%','90%','Maximum'])
actual = pd.DataFrame(mobility_la['date'],columns = ['date'])
predicted = pd.DataFrame(mobility_la['date'],columns = ['date'])
for i in counties:
  temp = df[df['fips']==i]
  temp.reset_index(inplace = True,drop = True)
  df_train = temp[0:int(0.8*len(temp))]
  df_test = temp[int(0.8*len(temp)):]
  #df_test['date'] = pd.to_datetime(df_test['date'])
  cases_train = df_train['cases'].tolist()
  cases_test = df_test['cases'].tolist()
  SEQUENCE_SIZE = 7
  x_train,y_train = to_sequences(SEQUENCE_SIZE,cases_train)
  x_test,y_test = to_sequences(SEQUENCE_SIZE,cases_test)
  if len(x_test) > 0:
  # print("Shape of training set: {}".format(x_train.shape))
  # print("Shape of test set: {}".format(x_test.shape))
    predictions = model.predict(x_test)
    diff = pd.DataFrame(predictions) - pd.DataFrame(y_test)
    diff = abs(diff)
    metrics.loc[len(metrics.index)] = [i,mse(predictions,y_test,squared = False),y_test.mean(),(mse(predictions,y_test,squared = False)/y_test.mean())*100,len(y_test),diff.values.min(),np.median(diff),np.quantile(diff,0.10),np.quantile(diff,0.25),np.quantile(diff,0.75),np.quantile(diff,0.90),diff.values.max()]#,np.sum(np.array(diff) < 0, axis=0)]
    date = pd.DataFrame(df_test.tail(-7)['date'])
    date.reset_index(inplace = True,drop = True)
    predictions = pd.concat([pd.DataFrame(predictions,columns = [i]),date],axis = 1)
    y_test = pd.concat([pd.DataFrame(y_test,columns = [i]),date],axis = 1)
    actual = actual.merge(y_test,on = ['date'],how = 'left')
    predicted = actual.merge(predictions,on = 'date',how = 'left')

  else:
    pass
  print(i)

metrics_path = '/path'
actual_path =  '/path'
predictions_path = '/path'

metrics.to_csv(metrics_path,index = False)
actual.to_csv(actual_path,index = False)
predicted.to_csv(predictions_path,index = False)
