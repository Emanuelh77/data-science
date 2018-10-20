from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
# this makes it so that I can get the same result over and over 
# with the same random numbers
numpy.random.seed(7)

# LOAD PIMA INDIANS DATASET
# Pima Indians onset of diabetes
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
# there are 8 input variables and one output variable in the csv file
X = dataset[:, 0:8]
Y = dataset[:, 8]

# CREATE MODEL
# models in Keras are defined as a sequence of layers
model = Sequential()
# Relu is used for better performance
# Fully connected layers are defined using the Dense class
model.add(Dense(12, input_dim=8, activation='relu'))    #1st layer, 12 neurons and 8 input variables
model.add(Dense(8, activation='relu'))                  #hidden layer, 8 neurons
model.add(Dense(1, activation='sigmoid'))               #output layer, 1 neuron, sigmoid used to ensure output is between 0 and 1

# COMPILE MODEL
# loss evaluates how well a network does with the given data
# adam is an efficient gradient descent algorithm
# classification problem so accuracy is used as the metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# FIT THE MODEL aka train the neural network
#epochs = # of times the training process with go for
#batch_size number is how often the weight is adjusted
model.fit(X, Y, epochs=25, batch_size=10)

# EVALUATE THE MODEL
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
#predictions = model.predict(X)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
