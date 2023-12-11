#%%
# build a binary classifier for speed dating prediction dataset using keras
# This neural network is l2 regularized with two hidden layers. There is also
# gradient clipping to prevent exploding gradients. The model is trained using
# cross validation. The final model is evaluated on the test set and the
# precision score is calculated.

# https://www.kaggle.com/datasets/ulrikthygepedersen/speed-dating

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score

df = pd.read_csv('speeddating.csv')

# Drop rows with race=='?', race_o=='?', met!='0.0' and met!='1.0'
df = df[df.race!="b'?'"]
df = df[df.race_o!="b'?'"]

df.met = df.met.astype(float)
df = df[df.met <= 1.0]

# drop rows with NaN values
df = df.dropna()
df = df.drop(['has_null'], axis=1)

for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].apply(lambda x: x.replace("b'", '')) # remove b'
        df[c] = df[c].apply(lambda x: x.replace("'", '')) # remove '

# turn the numerical columns into float
try:
    df = df.astype(float)
except ValueError:
    pass

# for some reason, the columns 'decision', 'decision_o', 'match' are not floats
for c in ['decision', 'decision_o', 'match']:
    df[c] = df[c].astype(float)

# perform one-hot encoding on categorical variables using pdf.get_dummies
df = pd.get_dummies(df)

# split the data into training and test sets
X = df.drop(['match', 'decision_o'], axis = 1)
y = df.match

# normalize the training and test sets
scaler = StandardScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1],
                                                kernel_regularizer=l2(0.01)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = RMSprop(clipnorm=1.0)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                                                        metrics=['accuracy'])
    return model

# perform cross validation
k = 5
num_val_samples = len(X_train) // k
all_mae_histories = []

for i in range(k):
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

    partial_X = np.concatenate([X_train[:i * num_val_samples],
                                X_train[(i+1) * num_val_samples:]], axis=0)
    partial_y = np.concatenate([y_train[:i * num_val_samples],
                            y_train[(i + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(partial_X, partial_y,
                        validation_data=(val_data, val_targets),
                        epochs=25, batch_size=64, verbose=0)
    mae_history = history.history["val_accuracy"]
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories])
                                                        for i in range(25)]

print('Cross validation accuracy: ', np.mean(average_mae_history))

model = build_model()
model.fit(X_train, y_train, epochs=15, batch_size=32)

# model.fit(X_train, y_train, batch_size=64, epochs=25, validation_split=0.2)
# use the above line if you don't want to use cross validation

print(model.evaluate(X_test, y_test))

y_pred = model.predict(X_test)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
print('precision: ', precision_score(y_test, y_pred))