'''
    Recomendation System using Boltzmann Machine
    Predict movie rating by a user
    yes or no system
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import logging
import os

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger(__name__)

# directory ml-1m contains all of our data movie
# directory ml-100k contains 5 partitions of our data movie split in train and test, each with 100k rows


# import dataset
logging.info('Reading movies')
movies = pd.read_csv('ml-1m/movies.dat', sep="::", header = None, engine = 'python', encoding = 'latin-1', usecols = [0,1,2])
users  = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1', usecols = [0,1,2,4])
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1', usecols = [0,1,2,3])

logging.info('Shape of movies dataframe: {}'.format(movies.shape))
logging.info('Shape of users dataframe: {}'.format(users.shape))
logging.info('Shape of ratings dataframe: {}'.format(ratings.shape))


# preparing the training set and test set
logging.info('Preparing the training set and test set')
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')

logging.info('Shape of training_set: {}'.format(training_set.shape))
logging.info('Shape of test_set: {}'.format(test_set.shape))

# convert into array
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies in training_set + test_set
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
logging.info('Total of users is: {}'.format(nb_users))
logging.info('Total of movies is: {}'.format(nb_movies))

# Converting the data into a array with users in lines and movies in columns
logging.info('Converting into array')
def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:,1] [data[:,0] == id_users]
        id_ratings = data[:,2] [data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# each row is a user and each column is movie rating from user
training_set = convert(training_set)
test_set = convert(test_set)
logging.info('Now, each row is a unique user and each column is a single movie with his rating')

# Converting the data into Torch tensors
logging.info('Converting into Torch tensors')
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (liked) or 0 (not liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# training the RBM
logging.info('Training RBM')

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# testing the RBM predicting

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
        s += 1.
print('test loss: ' + str(test_loss/s))
