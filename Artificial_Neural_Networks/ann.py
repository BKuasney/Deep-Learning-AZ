import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import pickle
import os

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger(__name__)
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('keras').setLevel(logging.WARNING)



'''
    Using a cnn algorithm to make churn predict with a bank daatbase
    Output:
        1 if the customer leaves the bank
        0 if the customer don't leave the bank
'''


'''
Step 1 - Data Preprocessing
'''

# Importing DataSet
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
logging.info('Import OK')

# Encoding categorical data (country and gender)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])

# create dummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# we need to remove one dummy varibale
X = X[:,1:]
logging.info('Dummy OK')

# Splitting the dataset into the Training Set and Teste set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 0)
logging.info('Create train and test OK')

# Feature Scaling
# all variables to the same scale (log, normal, etc)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
logging.info('Feature Scaling OK')


'''
    Part 2: let's make the ANN
'''

logging.info('initializing ANN...')
from keras.models import Sequential
from keras.layers import Dense

# initializing ANN
# Let's create a sequence of layers
classifier = Sequential()

# Trainning ANN with Stochastic Gradient Descent
    # Step1
        # Randomly initialise the weights to small numbers close to 0, but not 0
    # Step 2
        # Input the first observation of your dataset in the input layer, each feature in one input node
    # Step 3
        # Forward-Propagation: from left to right, the neurons are activated in a way that the impact of each
        # neuron's activation is limited by the weights. Propagate the activations until getting the prediced result
    # Step 4
        # Compare the predicted result to the actual result. Measure the generated error
    # Step 5
        # Back-propagation: from the righ to left, the error is back-propagated
        # Update the weights according to how much they responsible for the error
        # The learning rate decides by how much we update the weights
    # Step 6
        # Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning)
        # OR
        # Repeat Steps 1 to 5 bu update the weights only after a batch of observation (Batch Learning)
    # Step 7
        # When the whole training set passed throungh the ANN that makes an epoch. Redo more epochs



# IF MODEL DON'T EXIST THEN RUN, ELSE LOAD
from keras.models import model_from_json

if os.path.isfile('./ann_model.json') == True:
    # load json and create model
    json_file = open('ann_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights('model.h5')

    logging.info('Loaded Model from disk')


else:

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(
        output_dim = 6,
        # we have 11 independent variables + 1 dependent variable, so, take a avg (11+1/2 = 6 nodes)
        init = 'uniform',
        # stochastic gradient descent, randomly init with a number close to 0
        activation = 'relu',
        # activation function (sigmnoid, relu, etc)
        input_dim = 11
        # independent variables
        ))

    logging.info('first layer OK')

    # Adding the second hidden layer
    # remove input layer
    classifier.add(Dense(
        output_dim = 6,
        # we have 11 independent variables + 1 dependent variable, so, take a avg (11+1/2 = 6 nodes)
        init = 'uniform',
        # stochastic gradient descent, randomly init with a number close to 0
        activation = 'relu'
        # activation function (sigmnoid, relu, etc)
        ))

    # Adding the output layer
    # Add output = 1
    # change activation fucntion to sigmoid (probabilistic result)
    classifier.add(Dense(
        output_dim = 1,
        # we have 11 independent variables + 1 dependent variable, so, take a avg (11+1/2 = 6 nodes)
        init = 'uniform',
        # stochastic gradient descent, randomly init with a number close to 0
        activation = 'sigmoid'
        # activation function (sigmnoid, relu, etc)
        ))


    # Compiling the ANN
    classifier.compile(
        optimizer = 'adam',
        # Adam optimization algorithm is an extension to stochastic gradient descent
        # update network weights iterative based in training data.
        loss = 'binary_crossentropy',
        # if is binary then binary cross entropy
        metrics = ['accuracy']
        # method to evaluate the model
        )

    logging.info('Compiling Ann OK')

    # Fitting the ANN to the Training Set

    classifier.fit(
        X_train, y_train,
        batch_size = 10,
        # recalculate the weigts for each 10 interactions
        nb_epoch = 100
        # we recreate the experiment 100 x, extracting 10 batch size randomly
    )
    # after running, we obtained a accuracy about 83%

    # Serealize to JSON to save
    classifier_json = classifier.to_json()
    with open ('ann_model.json', 'w') as json_file:
        json_file.write(classifier_json)
    # Serialize weights into new model
    classifier.save_weights('model.h5')
    logging.info('Saved model to disk')


'''
    Part 3: Making the predictions and evaluating the model
'''

# Prediction the test and result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Makgin the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

logging.info(cm)
logging.info('accuracy: {}'.format((cm[0][0]+cm[1][1])/y_test.shape[0]))


'''
    Part 4: Tunning Ann
'''

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# make model
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

logging.info('Tunning...')

parameters = {'batch_size': [25,32],
              'nb_epoch': [100,500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10)

logging.info('tunning fit')

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

logging.info('Best Parameters:'.format(grid_search.best_params_))
logging.info('Best Accuracy:'.format(grid_search.best_score_))
