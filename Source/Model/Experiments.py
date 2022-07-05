"""
Author:         Landon Buell
Date:           July 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        Model
File:           PipelineTasks.py
"""

        #### IMPORTS ####

import numpy as np

import PipelineUtilities

        #### CLASS DEFINTIONS ####

class Experiment:
    """ Experiment isan Abstract Base Class for all Experiments """

    def __init__(self,outputPath,numIters=1):
        """ Constructor """
        self._outputPath    = outputPath
        self._queue         = PipelineUtilities.Pipeline(self)

        self._preCallbacks  = []
        self._postCallbacks = []

        self._designMatrix  = None
        self._targetLabels  = None

        self._numIters      = numIters
    
    def __del__(self):
        """ Destructor """

    # Getters and Setters

    def getOutputPath(self):
        """ Return the Output Path for this experiment """
        return self._outputPath

    def getQueue(self):
        """ Return the Queue of Tasks """
        return self._queue

    # Public Interface

    def registerPreprocessCallbacks(self,callback):
        """ Register a method to be evaluated before the pipeline """
        self._preCallbacks.append(callback)
        return self

    def registerPostprocessCallbacks(self,callback):
        """ Register a method to be evaluated after the pipeline """
        self._postCallbacks.append(callback)
        return self

    def run(self):
        """ Execute the Experiment """
        self.evaluatePreprocessCallbacks()
        self._queue.evaluate()
        self.evaluatePostprocessCallbacks()
        return self

    # Protected Interface
       
    def buildQueue(self):
        """ Construct the Queue of Tasks for this experiment """
        return self

    def evaluatePreprocessCallbacks(self):
        """ Evaluate Callbacks before the pipeline """
        for item in self._preCallbacks:
            item()
        return self

    def evaluatePostprocessCallbacks(self):
        """ Evaluate Callbacks after the pipeline """
        for item in self._postCallbacks:
            item()
        return self

class BaselineDigits28x28(Experiment):
    """ Baseline Experiment w/ 28x28 Digits """

    def __init__(self,outputPath,numIters):
        """ Constructor """
        super().__init__(outputPath,numIters)
        self.buildQueue()

    def __del__(self):
        """ Destructor """
        super().__del__()

    # Protected interface

    def buildQueue(self):
        """ Construct the Queue of Tasks for this experiment """
        self._queue.registerTask()

        return self