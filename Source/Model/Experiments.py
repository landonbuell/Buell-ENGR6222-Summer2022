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

import Managers

        #### CLASS DEFINTIONS ####

class Experiment:
    """ Experiment isan Abstract Base Class for all Experiments """

    def __init__(self,
                 outputPath,
                 datasetCode,
                 trainSize=0.8,
                 trainEpochs=100,
                 numIters=1):
        """ Constructor """
        self._numIters      = numIters

        self._datasetManager    = None      # Class to Loading in Dataset as (x,y) pair
        self._modelManager      = None      # Class to build Model based off of dataset
        self._trainManager      = None      # Class to use data to train the models
        self._rundataManager    = None      # Class to track runtime Information

        self._preCallbacks  = []
        self._postCallbacks = []

        self.registerDatasetManager(    Managers.DatasetManager(datasetCode)    )
        self.registerModelManager(      Managers.ConvNeuralNetworkBuilder(datasetCode)     )
        self.registerTrainingManager(   Managers.TrainingManager(trainSize,trainEpochs)     )
        self.registerRundataManager(    Managers.ExportManager(outputPath)      )

    def __del__(self):
        """ Destructor """
        self._datasetManager    = None      
        self._modelManager      = None    
        self._trainManager      = None   
        self._rundataManager    = None   

    # Getters and Setters

    def getOutputPath(self):
        """ Return the output Path for all data """
        return self.getRundataManager().getOutputPath()

    def getNumIterations(self):
        """ Return the Number of times to repeat this experiment """
        return self._numIters

    def getDatasetManager(self):
        """ Return the Dataset Manager Instance """
        if (self._datasetManager is None):
            errMsg = "ERROR: DatasetManager is None"
            raise RuntimeError(errMsg)
        return self._datasetManager

    def getModelManager(self):
        """ Return the Model Manager Instance """
        if (self._modelManager is None):
            errMsg = "ERROR: ModelManager is None"
            raise RuntimeError(errMsg)
        return self._modelManager

    def getTrainManager(self):
        """ Return the Train Manager Instance """
        if (self._trainManager is None):
            errMsg = "ERROR: TrainManager is None"
            raise RuntimeError(errMsg)
        return self._trainManager

    def getRundataManager(self):
        """ Return the Dataset Manager Instance """
        if (self._rundataManager is None):
            errMsg = "ERROR: RundataManager is None"
            raise RuntimeError(errMsg)
        return self._rundataManager

    def getTrainingData(self):
        """ Return a Ref to the training data """
        return (self._designMatrix[0],self._targetLabels[0])
    
    def getTestingData(self):
        """ Return a Ref to the testing data """
        return (self._designMatrix[1],self._targetLabels[1])

    def setTrainingData(self,X,y):
        """ Return a Ref to the training data """
        self._designMatrix[0] = X
        self._targetLabels[0] = y
        return self
    
    def setTestingData(self,X,y):
        """ Return a Ref to the testing data """
        self._designMatrix[1] = X
        self._targetLabels[1] = y
        return self

    # Public Interface

    def registerPreprocessCallbacks(self,callback):
        """ Register a method to be evaluated before the pipeline """
        self._preCallbacks.append(callback)
        return self

    def registerPostprocessCallbacks(self,callback):
        """ Register a method to be evaluated after the pipeline """
        self._postCallbacks.append(callback)
        return self

    def registerDatasetManager(self,manager):
        """ Register a Dataset Manager to this Instance """
        self._datasetManager = manager
        self._datasetManager.setOwner(self)
        return self

    def registerModelManager(self,manager):
        """ Register a Model Manager to this Instance """
        self._modelManager = manager
        self._modelManager.setOwner(self)
        return self

    def registerTrainingManager(self,manager):
        """ Register a Training Manager to this Instance """
        self._trainManager = manager
        self._trainManager.setOwner(self)
        return self

    def registerRundataManager(self,manager):
        """ Register a Rundata Manager to this Instance """
        self._rundataManager = manager
        self._rundataManager.setOwner(self)
        return self

    def run(self):
        """ Execute the Experiment """
        self.checkAllManagersAreNotNone()
        self.evaluatePreprocessCallbacks()

        # Main Experiment Body
        self._datasetManager.call()

        # Number of Times to repeat the experiment
        for iter in range(self._numIters):
            self._modelManager.call()


            

        self.evaluatePostprocessCallbacks()

        return self

    # Protected Interface
       

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

    def checkAllManagersAreNotNone(self):
        """ Ensure that All Managers are registered """
        x = self.getDatasetManager()
        x = self.getModelManager()
        x = self.getTrainManager()
        x = self.getRundataManager()
        return self

    # Static Methods for Preset Experiments

    @staticmethod
    def getBaselineDigits28x28(outputPath):
        """ Baseline Experiment w/ 28x28 MNIST Digits """
        experiment = Experiment(
            outputPath,
            'mnist784',
            trainSize=0.8,
            trainEpochs=100,
            numIters=1)
        return experiment

    @staticmethod
    def getBaselineDigits8x8(outputPath):
        """ Baseline Experiment w/ 8x8 MNIST Digits """
        experiment = Experiment(
            outputPath,
            'mnist64',
            trainSize=0.8,
            trainEpochs=100,
            numIters=1)
        return experiment
