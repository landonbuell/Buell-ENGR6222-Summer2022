"""
Author:         Landon Buell
Date:           July 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        ImageProcessing
File:           Interpolation.py
"""

        #### IMPORTS ####

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


        #### Downsizing Callbacks ####

def averagePoolStep2Stride1(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (2,2)
    windowStep = (1,1)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDesignMatrix()
    X = poolingLayer.call( X )
    experiment.setDesignMatrix(X)
    return None

def averagePoolStep2Stride2(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (2,2)
    windowStep = (2,2)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDesignMatrix()
    X = poolingLayer.call( X )
    experiment.setDesignMatrix(X)
    return None

def averagePoolStep3Stride1(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (3,3)
    windowStep = (1,1)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDesignMatrix()
    X = poolingLayer.call( X )
    experiment.setDesignMatrix(X)
    return None

def averagePoolStep3Stride2(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (3,3)
    windowStep = (2,2)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDesignMatrix()
    X = poolingLayer.call( X )
    experiment.setDesignMatrix(X)
    return None

def averagePoolStep3Stride2(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (3,3)
    windowStep = (2,2)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDesignMatrix()
    X = poolingLayer.call( X )
    experiment.setDesignMatrix(X)
    return None

        #### Export Images ####

def showImages(experiment,n=10):
    """ Show the first 'n' samples in the dataset """
    X = experiment.getDatasetManager().getDesignMatrix()
    for ii in range(0,n,1):
        plt.imshow(X[ii],cmap=plt.cm.gray)
        plt.show()
        plt.close()
    return None

def saveImages(experiment,n=10):
    """ Log the first 'n' samples in the dataset """
    X = experiment.getDatasetManager().getDesignMatrix()
    for ii in range(0,n,1):
        figName = "sample{0}.png".format(ii)
        outpath = os.path.join(experiment.getOutputPath(),figName)
        image = X[ii]
        plt.figure(figsize=(16,12),edgecolor='gray',tight_layout=True)
        plt.imshow(X[ii],cmap=plt.cm.gray)
        plt.savefig(outpath)
        plt.close()
    return None
