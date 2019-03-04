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
dataset = pd.read_csv('dataset/Social_Network_Ads.csv')
x = dataset.iloc[: [2,3]].values
y = dataset.iloc[: 4].values
logging.info(x)
logging.info(y)
