"""
Author:         Landon Buell
Date:           July 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        Model
File:           PipelineTasks.py
"""


        #### IMPORTS ####

import os
import sys

import Experiments
import Callbacks

        #### FUNCTION DEFINITIONS ####

def getBaselineDigits8x8(outputPath):
    """ Baseline Experiment w/ 8x8 MNIST Digits """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        trainEpochs=2,
        numIters=10)
    return experiment

def getBaselineDigits28x28(outputPath):
    """ Baseline Experiment w/ 28x28 MNIST Digits """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        trainEpochs=2,
        numIters=10)
    return experiment

def getCaseStudy1(outputPath):
    """ See outline for Case-Study 1 Notes """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        trainEpochs=2,
        numIters=10)
    # Register some callbacks
    #experiment.registerPreprocessCallbacks( DownSampling.showImages )
    #experiment.registerPreprocessCallbacks( DownSampling.saveImages )
    return experiment


def getCaseStudy2(outputPath):
    """ See outline for Case-Study 1 Notes """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        trainEpochs=2,
        numIters=10)
    # Register some callbacks
    #experiment.registerPreprocessCallbacks( DownSampling.showImages )
    #experiment.registerPreprocessCallbacks( DownSampling.saveImages )
    return experiment

def getCaseStudy3(outputPath):
    """ See outline for Case-Study 1 Notes """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        trainEpochs=2,
        numIters=10)
    # Register some callbacks
    #experiment.registerPreprocessCallbacks( DownSampling.showImages )
    #experiment.registerPreprocessCallbacks( DownSampling.saveImages )
    return experiment

def getCaseStudy4(outputPath):
    """ See outline for Case-Study 1 Notes """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        trainEpochs=2,
        numIters=10)
    # Register some callbacks
    #experiment.registerPreprocessCallbacks( DownSampling.showImages )
    #experiment.registerPreprocessCallbacks( DownSampling.saveImages )
    return experiment