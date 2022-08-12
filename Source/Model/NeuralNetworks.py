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

        #### RUNTIME CONSTANTS ####

METRICS = [ tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Accuracy() ]

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

    def getF1Score(self):
        """ Get the Array of F1 Scores """
        return 2 * (self._precision * self._recall) / (self._precision  + self._recall)

    def getIterationCount(self):
        """ Get the number of iterations """
        return self._iterCount

    def getMetricsDictionary(self):
        """ Get all metrics as a dictionary """
        resultMap = {
            "Accuracy":     self._accuracy,
            "Loss":         self._losses,
            "Precision":    self._precision,
            "Recall":       self._recall,
            "F1-Score":     self.getF1Score() }
        return resultMap

    # Public Interface

    def update(self,batchLog):
        """ Update the state of this class w/ a history object """
        self._losses    = np.append(self._losses,   batchLog['loss'])
        self._accuracy  = np.append(self._accuracy, batchLog['accuracy'])
        self._precision = np.append(self._precision,batchLog['precision'])
        self._recall    = np.append(self._recall,   batchLog['recall'])
        self._iterCount += 1
        return self

    def clear(self):
        """ Clear all Metric Data """
        self._losses    = np.array([],dtype=np.float32)
        self._accuracy  = np.array([],dtype=np.float32)
        self._precision = np.array([],dtype=np.float32)
        self._recall    = np.array([],dtype=np.float32)
        self._iterCount = 0
        return self
    
class TestResults:
    """ Store the Test Results for a model after training """

    def __init__(self):
        """ Constructor """
        self._outputs       = np.array([],dtype=np.int16)
        self._labels        = np.array([],dtype=np.int16)
        self._numClasses    = 0

    def __del__(self):
        """ Destructor """

    def getResultsDictionary(self):
        """ Get all Results as a dictionary """
        numClasses = self._outputs.shape[-1]
        resultMap = {   "Labels":       self._labels,
                        "Predicitions": np.argmax(self._outputs,axis=-1) }      
        for i in range(numClasses):
            key = "class_{0}".format(i)
            val = self._outputs[:,i]
            resultMap.update({key:val})
        return resultMap

    def update(self,predictions,labels):
        """ Update the testing results """
        self._outputs = predictions
        self._labels = labels
        return self

    def clear(self):
        """ Clear all output data """
        self._outputs       = np.array([],dtype=np.int16)
        self._labels        = np.array([],dtype=np.int16)
        return self

class NeuralNetworkModel:
    """ Class to build + Compile TF Neural network models """

    def __init__(self):
        """ Constructor """
        msg = "{0} is a static class. Cannot make instance".format(self.__class__)
        raise RuntimeError(msg)

    # Public Interface

    @staticmethod
    def getConvNeuralNetworkA(inputShape,numClasses,randomSeed=0):
        """ Generate and Compile a Tensorflow Convolutional Neural Network """
        tf.random.set_seed(randomSeed)
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.InputLayer(input_shape=inputShape)    )

        # 1st Layer Group
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(2,2),activation='relu') )
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(2,2),activation='relu') )
        model.add( tf.keras.layers.MaxPooling2D(pool_size=(3,3)) ) 

        # Multilayer Perceptron
        model.add( tf.keras.layers.Flatten() )
        model.add( tf.keras.layers.Dense(units=32,activation='relu') )
        model.add( tf.keras.layers.Dense(units=32,activation='relu') )
        model.add( tf.keras.layers.Dense(units=numClasses,activation='softmax') )

        # Compile
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=METRICS )
        return model

    @staticmethod
    def getConvNeuralNetworkB(inputShape,numClasses,randomSeed=0):
        """ Generate and Compile a Tensorflow Convolutional Neural Network """
        tf.random.set_seed(randomSeed)
        model = tf.keras.models.Sequential()
        model.add( tf.keras.layers.InputLayer(input_shape=inputShape)    )

        # 1st Layer Group
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu') )
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu') )
        model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) ) 

        # 2nd Layer Group
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu') )
        model.add( tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu') )
        model.add( tf.keras.layers.MaxPooling2D(pool_size=(2,2)) ) 

        # Multilayer Perceptron
        model.add( tf.keras.layers.Flatten() )
        model.add( tf.keras.layers.Dense(units=64,activation='relu') )
        model.add( tf.keras.layers.Dense(units=64,activation='relu') )
        model.add( tf.keras.layers.Dense(units=32,activation='relu') )
        model.add( tf.keras.layers.Dense(units=32,activation='relu') )
        model.add( tf.keras.layers.Dense(units=numClasses,activation='softmax') )

        # Compile
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=METRICS)
        return model

    