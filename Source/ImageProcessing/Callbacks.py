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

import Interpolation

        #### Downsizing Callbacks ####

def averagePoolSize2Stride1(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (2,2)
    windowStep = (1,1)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDatasetManager().getDesignMatrix()
    print("X.shape -> ",X.shape)
    X = poolingLayer.call( X )
    print("X.shape -> ",X.shape)
    experiment.getDatasetManager().setDesignMatrix(X)
    return None

def averagePoolSize2Stride2(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (2,2)
    windowStep = (2,2)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDesignMatrix()
    print("X.shape -> ",X.shape)
    X = poolingLayer.call( X )
    print("X.shape -> ",X.shape)
    experiment.setDesignMatrix(X)
    return None

def averagePoolSize3Stride1(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (3,3)
    windowStep = (1,1)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDesignMatrix()
    print("X.shape -> ",X.shape)
    X = poolingLayer.call( X )
    print("X.shape -> ",X.shape)
    experiment.setDesignMatrix(X)
    return None

def averagePoolSize3Stride2(experiment):
    """ Down Sample Input Images using 2 x 2 pooling  w/ 1 x 1 step"""
    windowSize = (3,3)
    windowStep = (2,2)
    poolingLayer = tf.keras.layers.AveragePooling2D(
        pool_size=windowSize,strides=windowStep)
    X = experiment.getDesignMatrix()
    print("X.shape -> ",X.shape)
    X = poolingLayer.call( X )
    print("X.shape -> ",X.shape)
    experiment.setDesignMatrix(X)
    return None

        #### Interpolation Callbacks ####

def interpolateBilinear(experiment):
    """ Perform Bilinear Interpolation """
    numSamples = experiment.getDatasetManager().getNumSamples()
    inputShape = experiment.getDatasetManager().getInputShape()
    currShape = experiment.getDatasetManager().getDesignMatrix().shape

    # Existing Design Matrix + Empty one to populate
    X = experiment.getDatasetManager().getDesignMatrix()
    Y = experiment.getDatasetManager().buildEmptyDesignMatrix()

    # Build up data for interpolation 
    xAxisOld = np.linspace(0,inputShape[0],currShape[1])
    yAxisOld = np.linspace(0,inputShape[1],currShape[2])
    xAxisNew = np.linspace(0,inputShape[0],inputShape[0])
    yAxisNew = np.linspace(0,inputShape[1],inputShape[1])

    # Peform Interpolation
    for i in range(numSamples):
        x = X[i]
        interpFunc = Interpolation.interpolate.interp2d(
            xAxisOld,yAxisOld,x,kind='linear',copy=False)   # Get interp function
        y = interpFunc(xAxisNew,yAxisNew).reshape(inputShape)
        Y[i] = y
    # Assign X back to Design Matrix
    experiment.getDatasetManager().setDesignMatrix(Y)
    X = None
    return None

        #### Export + Show Images ####

def showImages(experiment,n=10):
    """ Show the first 'n' samples in the dataset """
    X = experiment.getDatasetManager().getDesignMatrix()
    for ii in range(0,n,1):
        plt.figure(figsize=(16,12),edgecolor='gray',tight_layout=True)
        plt.imshow(X[ii],cmap=plt.cm.gray)
        plt.xticks(np.arange(X[ii].shape[0]))
        plt.yticks(np.arange(X[ii].shape[1]))
        plt.show()
        plt.close()
    return None

def saveImages(experiment,name="baseline",n=4):
    """ Log the first 'n' samples in the dataset """
    X = experiment.getDatasetManager().getDesignMatrix()
    for ii in range(0,n,1):
        figName = "{0}{1}.png".format(name,ii)
        outpath = os.path.join(experiment.getOutputPath(),figName)
        plt.figure(figsize=(16,12),edgecolor='gray',tight_layout=True)
        plt.imshow(X[ii],cmap=plt.cm.gray)
        plt.xticks(np.arange(X[ii].shape[0]))
        plt.yticks(np.arange(X[ii].shape[1]))
        plt.savefig(outpath)
        plt.close()
    return None

def saveImageBaseline(experiment,n=4):
    """ Save first 'n' samples w/ baseline tag """
    return saveImages(experiment,"baseline",n)

def saveImageDownsized(experiment,n=4):
    """ Save first 'n' samples w/ baseline tag """
    return saveImages(experiment,"downsized",n)

def saveImageUpscaled(experiment,n=4):
    """ Save first 'n' samples w/ baseline tag """
    return saveImages(experiment,"interpolated",n)
