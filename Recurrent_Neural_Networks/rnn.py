'''
    Predictin Google's Stock Price
'''

# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os

# configure path
script_dir = os.path.dirname(__file__)

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger(__name__)
logging.getLogger('sklearn').setLevel(logging.CRITICAL)
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('keras').setLevel(logging.CRITICAL)


###############################
# Part 1 - Data Preprocessing #
###############################
logging.info("Start Part 1: Data Preprocessing")

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values # numpy array

# Feature Scaling
# make a normalisation: (x-min(X))/(max(X)-min(X))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # values between 0 and 1
training_set_scale = sc.fit_transform(training_set)

# creating a data sctructure with a 60 timesteps (1 window of 60 timesteps) and 1 output
# memorising what happens in 60 timesteps to predict timesteps 61
X_train = []
y_train = []

for i in range(60,1258):
    X_train.append(training_set_scale[i-60:i,0]) # 60 timesteps (t-60,t-59,...,t-n)
    y_train.append(training_set_scale[i,0]) # t

# X_train = each LINE is a window with 60 timesteps
# y_train = each LINE is a target to predict the next time (is the 61 timesteps)
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping to make a sctructure for RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

###########################
# Part 2 - Building a RNN #
###########################
logging.info("Start Part 2: Building a RNN")

# importing Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initialising RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(
                    units = 50, # higher numbers cause a higher dimensionality in a layer, but, small number don't capture standarts very well
                    return_sequences = True,
                    input_shape = (X_train.shape[1],1)
                    ))

regressor.add(Dropout(0.2)) # Dropout about 20% layers. So with 50 neurons in our rnn, we dropout 10 neurons

# Adding a second LSTM Layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM Layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM Layer and some Dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# Fully conect layer (used in previous codes)
regressor.add(Dense(units = 1)) # number of neuron to output. 1 output to predict the next value only

# Compile RNN
logging.info('Compiling RNN')
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Adam is a stochastic gradient descent apply to everything basically. Always use Adam for a first attempt
# RMSprop ia a stochastic gradient descent for RNN

# Fitting the RNN to the training set
logging.info('Fitting RNN')
regressor.fit(X_train, y_train, epochs = 100,
            batch_size = 32) # update the weights each 32 timesteps

logging.info('Saving Model')
# Save Model
model_backup_path = os.path.join(script_dir,'rnn_stock_price_google.h5')
regressor.save(model_backup_path)
logging.info("Saved")

###############################################################
# Part 3 - Making the predictions and visualising the results #
###############################################################
logging.info('Start Step 3: Predictions and Results')
logging.info('Making Data Preprocessing to predict')

# Getting the real stock price for 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock prices of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # vertical concatenation
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # lower bounds
# Scale
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# we train our model using all information, since 5 years behind
# now, we use only data from 80 days to predict the next stock price
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

# Reshpaing into a 3D format for RNN
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predict
logging.info('Making predictions')
predicted_stock_price = regressor.predict(X_test)

# rescale
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
logging.info('Predict OK')

# visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
