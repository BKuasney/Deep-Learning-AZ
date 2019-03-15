# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        # first full connection
        self.fc1 = nn.Linear(nb_movies, 20) # 20 nodes in the first hidden layer, detect 20 features. From nb_users nodes to 20 nodes in first hidden layer
        # second full connection
        self.fc2 = nn.Linear(20, 10) # from 20 nodes in first hidden layer to 10 nodes in second hidden layer
        # third full connection
        self.fc3 = nn.Linear(10, 20) # End point to code and Initial point to decode
        # fourth full connection
        self.fc4 = nn.Linear(20, nb_movies) # deconding to original numbers of observations
        # activation function to activate neurons
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # move nb_users to 20 nodes
        x = self.activation(self.fc1(x)) # first encoding vector
        x = self.activation(self.fc2(x)) # second encoding vector
        x = self.activation(self.fc3(x)) # third encoding vector
        x = self.fc4(x) # decoding
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

sae = SAE()
criterion = nn.MSELoss() # loss function
# make a gradient descent to update the weights
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # like 'adam' optimizer, learning rate 0.01

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.  # float
    for id_user in range(nb_users):
        # input vector of features with all rating by the specific user
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone() # copy/clone input
        # optimize the memory (processing)
        if torch.sum(target.data > 0) > 0: # if contains at least one rating
            output = sae(input) # output variable predict rating
            target.require_grad = False # don't compute gradient with target (save memory/computation)
            output[target == 0] = 0
            loss = criterion(output, target)  # compute loss error
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # mean of the error
            loss.backward() # direction of update
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step() # intention of update
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))
