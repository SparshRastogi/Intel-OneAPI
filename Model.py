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

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv')
df = df[df['fips'] == 6037.0]
df.reset_index(inplace = True,drop = True)

df_train = df[0:int(0.8*len(df))]
df_test = df[int(0.8*len(df)):]

cases_train = df_train['cases'].tolist()
cases_test = df_test['cases'].tolist()

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


SEQUENCE_SIZE = 7
x_train,y_train = to_sequences(SEQUENCE_SIZE,cases_train)
x_test,y_test = to_sequences(SEQUENCE_SIZE,cases_test)

print("Shape of training set: {}".format(x_train.shape))
print("Shape of test set: {}".format(x_test.shape))


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-8)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-8)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=512,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=16,
    mlp_units=[64],
    mlp_dropout=0.2,
    dropout=0.25,
)

model.compile(
    optimizer='Adam',
    loss='mse',
    metrics=[keras.metrics.RootMeanSquaredError()])

model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=25, \
    restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)

cl = pd.concat([pd.DataFrame(y_test,columns = ['Original']),pd.DataFrame(model.predict(x_test),columns = ['Predicted'])],axis = 1)

cl.index = df_test.tail(-7).index
cl['date'] = df_test['date']


import plotly.express as px
fig = px.line(cl,x = 'date',y = ['Predicted','Original'])
fig.show()

cl['Difference'] = cl['Original'] - cl['Predicted']

error = (mse(model.predict(x_test),y_test,squared = False)/y_test.mean())*100
model_path = '/'
model.save(model_path)
