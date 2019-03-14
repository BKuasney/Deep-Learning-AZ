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
# first columns of users is user_id, second column is gender, third column is age, fourth column is the user_job
# first columns of ratings is users (equal numbers means same user), second color correspond to movie_id, the third colum correspond to the ratings (1 to 5)
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
# in original dataframe, 0 means the user don't watch de movie, so, we have to transform 0 in -1
training_set[training_set == 0] = -1 # don't watch
training_set[training_set == 1] = 0  # not liked
training_set[training_set == 2] = 0  # not liked
training_set[training_set >= 3] = 1  # liked

test_set[test_set == 0] = -1 # don't watch
test_set[test_set == 1] = 0  # not liked
test_set[test_set == 2] = 0  # not liked
test_set[test_set >= 3] = 1  # liked

# Creating the architecture of the Neural Network
class RBM():
    # initialize the future object
    def __init__(self, nv, nh): # nv = visible nodes and nh = hidden nodes
        self.W = torch.randn(nh,nv) # weights using normal distribution
        self.a = torch.randn(1, nh) # batch and bias
        self.b = torch.randn(1, nv)

    def sample_h(self, x):
    # sample weith function -> sigmoid activation function
        wx = torch.mm(x, self.W.t()) # product Wx (transpose)
        activation = wx + self.a.expand_as(wx) # wx + bias_hidden_node = activation_function ("expand_as" is to apply bias for each line of the mini batch)
        p_h_given_v = torch.sigmoid(activation) # probability the hidden node is activate given the value of the visible node
        return p_h_given_v, torch.bernoulli(p_h_given_v)
        # return a sample of weights of hidden neurons. For example, if p_h_given_v is a probability like 0.7. If the random number is below of 0.7 then activate the neuron. if this number is more larger then not activate the neuron
        # we use a randon number to verify if our estimatives is better than a random number

    def sample_v(self, y):
    # return the probability of user liked or not liked the movie that user don't watch
    # same function but, to visible nodes
        wy = torch.mm(y, self.W) # product Wx
        activation = wy + self.b.expand_as(wy) # wx + bias_visible_nodes = activation_function ("expand_as" is to apply bias for each line of the mini batch)
        p_v_given_h = torch.sigmoid(activation) # probability the hidden node is activate given the value of the visible node
        return p_v_given_h, torch.bernoulli(p_v_given_h)
        # return a sample of weights of hidden neurons. For example, if p_h_given_v is a probability like 0.7. If the random number is below of 0.7 then activate the neuron. if this number is more larger then not activate the neuron
        # we use a randon number to verify if our estimatives is better than a random number

    def train(self, v0, vk, ph0, phk):
    # implement the algoritm from article Boltzmann-Machines
    # v0 is the input vector (all the rating by one user) then make a loop for all users
    # vk is visible k nodes
    # ph0 is de probability of first iteration the hidden nodes equal 1 given the value v0
    # phk correspond the probabilities of hidden nodes after k sampling  given the values of visible nodes vk
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t() # update weights with product of 2 tensors
        self.b += torch.sum((v0 - vk), 0) # update bias visible nodes
        self.a += torch.sum((ph0 - phk), 0) # update bias hidden nodes. prbability p_h_given_v

nv = len(training_set[0])
nh = 100 # choose any number of dataframe... identify 100 features
batch_size = 100 # update of n observation
rbm = RBM(nv, nh)

# training the RBM
logging.info('Training RBM')

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0 # add loss variable to predict loss between predicted and real (both values between 0 and 1)
    s = 0. # counting to normalize the loss (with float format)
    for id_user in range(0, nb_users - batch_size, batch_size): # update earch batch and not each user
        vk = training_set[id_user:id_user+batch_size] # k steps of randon walking (from id_user up to next 100 users)
        v0 = training_set[id_user:id_user+batch_size] # rating movie
        ph0,_ = rbm.sample_h(v0) # initial probability (return p_h_given_v) (probability of hidden layer given visible layer) (ph0,__ return only the first augment)
        for k in range(10): # k steps of contrast divergence (k steps of random walkings)
            _,hk = rbm.sample_h(vk) # (_,hk returns the second output of function)
            _,vk = rbm.sample_v(hk) # update vk
            vk[v0<0] = v0[v0<0]

        # start training
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    logging.info('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))

# testing the RBM predicting

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
logging.info('test loss: ' + str(test_loss/s))
