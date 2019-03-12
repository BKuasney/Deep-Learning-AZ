'''
    Make a hybrid deep learning model
    1 - identify fraud using Self-Organizing Maps
    2 - Going from unsupervised learning to Supervised Deep Learning
'''

# SOM CODES RUN ONLY ONCE!!!
# EACH RUNNING MODIFY THE LOCATION OF BETTER NODES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

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


'''
    First Step: Build a Self-Organizing Maps
'''
logging.info('Building Self_Organizing Maps')

# read data frame
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# training SOM
# file minison is a "base code" to run the som project. Like cascade is a "base code" to identify objects
logging.info('Importing minison')
from minisom import MiniSom

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
logging.info('Training...')
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) # transpose of matrix of all distances
colorbar()
markers = ['o','s']
colors = ['r','g']

for i, x in enumerate(X):
    w = som.winner(x) # winner node of customers x
    plot(
        w[0] + 0.5, w[1] + 0.5, # coordenate to the center of square
        markers[y[i]], # use markers 'o' if aprovall and 's' if not aprovall
        markeredgecolor = colors[y[i]], # same for colors
        markerfacecolor = 'None',
        markersize = 10,
        markeredgewidth = 2
        )

show()

# Finding the frauds
mappings = som.win_map(X) # list all the nodes associate, node (0,0) indicates the first up-left corner in the chart
frauds = np.concatenate((mappings[(5,3)], mappings[(8,3)]), axis = 0)
logging.info(frauds)
frauds = sc.inverse_tranform(frauds) # id of potential customers to fraud

'''
    Part 2: ANN
'''

# creating the matrix features
customers = dataset.iloc[:,1:].values # all the columsn excpet the last column

# creating the depend variable
is_fraud = np.zeros(len(dataset)) # vector of zeros with len of our dataset

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds: # if customer id is in a list fraud
        is_fraud[i] = 1 # replace all zeros for 1 if is a fraud

'''
    Making ANN
'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

# initializing ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu', input_dim = 15))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set
classifier.fit(customers, is_fraud, batch_size = 1, nb_epoch = 1)

# Predicting the probability of frauds
y_pred = classifier.predict(customers)
y_pred = pd.concatenate((dataset.iloc[:,0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()] # sort by column 1
