"""
Author:         Landon Buell
Date:           July 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        Model
File:           NeuralNetworks.py
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

        #### CLASS DEFINITIONS ####

class RunningHistory:
    """ Store the history of a model over multiple epochs """

    def __init__(self):
        """ Constructor """
        self._losses    = np.array([],dtype=np.float32)
        self._accuracy  = np.array([],dtype=np.float32)
        self._precision = np.array([],dtype=np.float32)
        self._recall    = np.array([],dtype=np.float32)
        self._iterCount = 0

    def __del__(self):
        """ Destructor """

    # Getters and Setters

    def getLosses(self):
        """ Get the array of losses """
        return self._losses

    def getAccuracies(self):
        """ Get the array of accuracies """
        return self._accuracy

    def getPrecisions(self):
        """ Get the array of precisions """
        return self._precision

    def getRecall(self):
        """ Get the array of recalls """
        return self._recall

    def getIterationCount(self):
        """ Get the number of iterations """
        return self._iterCount

    # Public Interface

    def update(self,history):
        """ Update the state of this class w/ a history object """
        self._losses    = np.append(self._losses,history['loss'])
        self._accuracy  = np.append(self._accuracy,history['accuracy'])
        self._precision = np.append(self._precision,history['precision'])
        self._recall    = np.append(self._recall,history['recall'])
        self._iterCount += 1
        return self

    def plotHistory(self,show=True,save=None):
        """ Generate a plot of history """
        plt.figure(figsize=(16,12),facecolor='gray')
        plt.title()
    

class NeuralNetworkModel:
    """ Class to build + Compile TF Neural network models """

    def __init__(self):
        """ Constructor """
        msg = "{0} is a static class. Cannot make instance".format(self.__class__)
        raise RuntimeError(msg)

    # Public Interface

    def getConvNeuralNetwork(self,inputShape,numClasses):
        """ Generate and Compile a Tensorflow Convolutional Neural Network """
        model = tf.keras.models.Sequential()

        # 1st Layer Group
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=inputShape) )
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu') )
        model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) ) 

        # 2nd Layer Group
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu') )
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu') )
        model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) ) 

        # Multilayer Perceptron
        model.add( tf.keras.layers.Flatten() )
        model.add( tf.keras.layers.Dense(units=32,activation='relu') )
        model.add( tf.keras.layers.Dense(units=32,activation='relu') )
        model.add( tf.keras.layers.Dense(units=numClasses,activation='softmax') )

        # Compile
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy','precision','recall'])
        return model