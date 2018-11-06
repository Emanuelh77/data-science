from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#LOADING DATA SET
white = pd.read_csv("winequality-white.csv", sep=";")
red = pd.read_csv("winequality-red.csv", sep=";")

red['type'] = 1
white['type'] = 2

wines = red.append(white, ignore_index=True)

X = wines.ix[:, 0:11]
y = np.ravel(wines.type)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the scaler
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

# Add an output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

#Predictions
y_pred = model.predict(X_test)
array([[0],
       [1],
       [0],
       [0],
       [0]], dtype=int32)

y_test[:5]
array([0, 1, 0, 0, 0])

score = model.evaluate(X_test, y_test, verbose=1)
print(score)



