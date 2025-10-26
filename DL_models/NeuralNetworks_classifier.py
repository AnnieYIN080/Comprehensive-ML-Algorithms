
import pandas as pd

concrete = pd.read_csv('../input/data.csv')
df = concrete.copy()

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

X_train = df_train.drop('Price', axis=1)
X_valid = df_valid.drop('Price', axis=1)
y_train = df_train['Price']
y_valid = df_valid['Price']

input_shape = [X_train.shape[1]]      
# the count of columns of X_train, the output of shape is count of [row, column]


#=============== Pls choose the suitable method, here is an example for famous methods to optimize the performance ==============

from tensorflow import keras
from tensorflow.keras import layers, callbacks

# increase the capacity of a network to avoid underfitting
# either by making it wider (more units to existing layers - linear) 
# or by making it deeper (adding more layers - nonlinear)
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),    # Adding Batch Normalization acts as a kind of adaptive preprocessor
    layers.Dense(512, activation='relu'),    # activation function: layers.Dense(512, activation='relu', input_shape = [11]),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3), 
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3), 
    layers.Dense(1, activation = 'sigmoid)
])

# To avoid too much noice (overfitting)
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

# "loss function" and "optimizer"
model.compile(
    optimizer='adam',        # Adam - "self tuning" - is an SGD algorithm that has an adaptive learning rate
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# "loss function" and "optimizer"
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=1000,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0,  # turn off training log
)

# plot for loss and val_loss
history_df = pd.DataFrame(history.history)

# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))

