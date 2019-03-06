'''
    1 - convolution
    2 - Pooling
    3 - Flattening
    4 - Full Conection
    5 - Fit
'''


import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import pickle
import os

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    # 32 features filters using a matrix 3x3
    # resize image to 64x64x3 (colored image)

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # pooling using MaxPooling to a matrix 2x2
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    # Add another convolutional layer to make our model more "deeper"
    # don't need a "input_shape" parameter on a second layer
    # we can make a 32, 64, etc feature filter
classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Add another pooling

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
    # turns matrix into a flat array

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
    # 128 input nodes to identify characteristics
    # 1 output (dog or cat, 0 or 1)

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # adam optimizer. Stochastic Gradient Descent

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, # pixels between 0 and 1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), # image resize to 64x64 (on a gpu we can put 128x128)
                                                 batch_size = 32, # batch
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64), # image resize to 64,64
                                            batch_size = 32, # batch
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # number of images on a training set
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000 # number of images on a test_set
                         )
