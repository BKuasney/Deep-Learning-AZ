import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger(__name__)


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
