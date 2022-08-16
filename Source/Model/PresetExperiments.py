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
        numIters=10)
    return experiment

def getBaselineDigits28x28(outputPath):
    """ Baseline Experiment w/ 28x28 MNIST Digits """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        numIters=10)
    experiment.registerPreprocessCallback( Callbacks.saveImageBaseline )
    return experiment

def getCaseStudy1(outputPath):
    """ See outline for Case-Study 1 Notes """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        numIters=10)
    # Register some callbacks
    experiment.registerPreprocessCallback( Callbacks.saveImageBaseline )
    experiment.registerPreprocessCallback( Callbacks.averagePoolSize2Stride1 )
    experiment.registerPreprocessCallback( Callbacks.saveImageDownsized)
    experiment.registerPreprocessCallback( Callbacks.interpolateLinearSpline )
    experiment.registerPreprocessCallback( Callbacks.saveImageUpscaled )
    return experiment


def getCaseStudy2(outputPath):
    """ See outline for Case-Study 1 Notes """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        numIters=10)
    # Register some callbacks
    experiment.registerPreprocessCallback( Callbacks.saveImageBaseline )
    experiment.registerPreprocessCallback( Callbacks.averagePoolSize2Stride2 )
    experiment.registerPreprocessCallback( Callbacks.saveImageDownsized)
    experiment.registerPreprocessCallback( Callbacks.interpolateLinearSpline )
    experiment.registerPreprocessCallback( Callbacks.saveImageUpscaled )
    return experiment

def getCaseStudy3(outputPath):
    """ See outline for Case-Study 1 Notes """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        numIters=10)
    # Register some callbacks
    experiment.registerPreprocessCallback( Callbacks.saveImageBaseline )
    experiment.registerPreprocessCallback( Callbacks.averagePoolSize3Stride1 )
    experiment.registerPreprocessCallback( Callbacks.saveImageDownsized)
    experiment.registerPreprocessCallback( Callbacks.interpolateLinearSpline )
    experiment.registerPreprocessCallback( Callbacks.saveImageUpscaled )
    return experiment

def getCaseStudy4(outputPath):
    """ See outline for Case-Study 1 Notes """
    outputPath = os.path.join("..\\..\\outputs",outputPath)
    experiment = Experiments.Experiment(
        outputPath,
        'mnist784',
        trainSize=0.8,
        numIters=10)
    # Register some callbacks
    experiment.registerPreprocessCallback( Callbacks.saveImageBaseline )
    experiment.registerPreprocessCallback( Callbacks.averagePoolSize3Stride2 )
    experiment.registerPreprocessCallback( Callbacks.saveImageDownsized)
    experiment.registerPreprocessCallback( Callbacks.interpolateLinearSpline )
    experiment.registerPreprocessCallback( Callbacks.saveImageUpscaled )
    return experiment