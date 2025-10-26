
import pandas as pd

concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
df = concrete.copy()

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

X_train = df_train.drop('CompressiveStrength', axis=1)
X_valid = df_valid.drop('CompressiveStrength', axis=1)
y_train = df_train['CompressiveStrength']
y_valid = df_valid['CompressiveStrength']

input_shape = [X_train.shape[1]]      # the count of columns of X_train 


#=============== Pls choose the suitable method, here is an example for famous methods to optimize the performance ==============

from tensorflow import keras
from tensorflow.keras import layers, callbacks
'''
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)
'''

model.compile(
    optimizer='sgd', # Stochastic Gradient Descent is more sensitive to differences of scale
    loss='mae',
    metrics=['mae'],
)

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(512, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mae'
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=50,
   # callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
