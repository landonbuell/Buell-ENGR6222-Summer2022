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
import numpy as np
import sklearn.datasets

import Managers
import NeuralNetworks

import DownSampling

        #### FUNCTION DEFINITIONS ####

@staticmethod
def loadSklearnDigits8x8(datasetManager):
    """ Load 8x8 digits from Sklearn data set """
    datasetBunch = sklearn.datasets.load_digits()
    bunchStruct = Managers.DatasetManager.DatasetBunchStruct()
    
    bunchStruct.designMatrix    = datasetBunch.images.reshape(1797,8,8,1)
    bunchStruct.targetVector    = datasetBunch.target
    bunchStruct.classNames      = list(datasetBunch.target_names)
    bunchStruct.numClasses      = len(datasetBunch.target_names)
    bunchStruct.description     = datasetBunch.DESCR

    datasetManager.registerDatasetBunch( bunchStruct )
    return None

@staticmethod
def loadSklearnDigits28x28(datasetManager):
    """ Load 28 x 28 digits from Sklearn Data set """
    datasetBunch = sklearn.datasets.fetch_openml("mnist_784",)
    bunchStruct = Managers.DatasetManager.DatasetBunchStruct()
    
    bunchStruct.designMatrix    = datasetBunch.data.to_numpy().reshape(70000,28,28,1)
    bunchStruct.targetVector    = datasetBunch.target.to_numpy(dtype=np.int16)
    bunchStruct.classNames      = list(np.unique(bunchStruct.targetVector))
    bunchStruct.numClasses      = len(bunchStruct.classNames)
    bunchStruct.description     = datasetBunch.DESCR

    datasetManager.registerDatasetBunch( bunchStruct )
    return None


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
        self._outputPath    = outputPath
        self._datasetCode   = datasetCode
        self._trainSize     = trainSize
        self._trainEpochs   = trainEpochs
        self._numIters      = numIters
        self._description   = ""
        

        self._datasetManager    = None      # Class to Loading in Dataset as (x,y) pair
        self._modelManager      = None      # Class to build Model based off of dataset
        self._trainManager      = None      # Class to use data to train the models
        self._rundataManager    = None      # Class to track runtime Information

        # Each callback will take a ref to the experiment as an argument
        self._preCallbacks  = []
        self._postCallbacks = []

        

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

    def registerDescription(self,descriptionString):
        """ Register a string to describe this experiment """
        self._description = descriptionString
        return self

    # Public Interface

    def initialize(self):
        """ Initialize Experiment for Running """
        self.registerDatasetManager(    Managers.DatasetManager(self._datasetCode)    )
        self.registerModelManager(      Managers.ConvNeuralNetworkBuilder(self._datasetCode)     )
        self.registerTrainingManager(   Managers.TrainingManager(self._trainSize,self._trainEpochs)     )
        self.registerRundataManager(    Managers.ExportManager(self._outputPath)      )
        return self

    def run(self):
        """ Execute the Experiment """
        self.checkAllManagersAreNotNone()
        
        # Main Experiment Body
        self._datasetManager.loadDataset()
        self.evaluatePreprocessCallbacks()
        seeds = np.random.randint(0,1e6,size=(self._numIters,))

        # Number of Times to repeat the experiment
        for i in range(self._numIters):

            # Generate the Model w/ a Random Seed
            self._modelManager.setSeed(seeds[i])
            self._modelManager.buildModel()

            # Split the Dataset 
            self._datasetManager.preprocessDataset()
            (X_train,X_test,y_train,y_test) = \
                self._trainManager.splitTrainTest()

            # Train + Evaluate
            self._trainManager.trainModel(X_train,y_train,i)
            self._trainManager.testModel(X_test,y_test,i)

            # Clear Everything
            self._rundataManager.clearTrainingHistory()
            self._rundataManager.clearTestingResults()

        # Output Run Information for Post-analysis
        self.evaluatePostprocessCallbacks()

        return self

    def exportConfiguration(self):
        """ Export experiment configuration details """
        outputPath = os.path.join(self._rundataManager.getOutputPath(),"config.txt")
        print("Exporting Configuration to '{0}'...".format(outputPath))
        strFmt = lambda x,y : "{0:<32}{1:<128}\n".format(x,y)
        with open(outputPath,"w") as outFile:
            outFile.write( "#"*64  + "\n")
            outFile.write( strFmt("OutputPath",    self._rundataManager.getOutputPath()) )
            outFile.write( strFmt("DatasetKey",    self._datasetManager.getDatasetKeyWord()) )
            outFile.write( strFmt("TrainSize",     self._trainManager.getTrainSize()) )
            outFile.write( strFmt("TestSize",      self._trainManager.getTestSize()) )
            outFile.write( strFmt("NumEpochs",     self._trainManager.getNumEpochs()) )
            outFile.write( strFmt("NumIters",      self._numIters) )
            outFile.write( strFmt("NumSamples",    self._datasetManager.getNumSamples()) )
            outFile.write( strFmt("NumClasses",    self._datasetManager.getNumClasses()) )
            outFile.write( "#"*64 + "\n")
        return self


    # Protected Interface
       
    def evaluatePreprocessCallbacks(self):
        """ Evaluate Callbacks before the pipeline """
        for item in self._preCallbacks:
            item.__call__(self)
        return self

    def evaluatePostprocessCallbacks(self):
        """ Evaluate Callbacks after the pipeline """
        for item in self._postCallbacks:
            item.__call__(self)
        return self

    def checkAllManagersAreNotNone(self):
        """ Ensure that All Managers are registered """
        x = self.getDatasetManager()
        x = self.getModelManager()
        x = self.getTrainManager()
        x = self.getRundataManager()
        return self

    # Static Methods for Preset Experiments
  
    



class DatasetHandler:
    """ Abstract Base Class to Handle + Parse Out datasets """

    DATASET_KEYWORD_TO_LOADER_CALLBACK = \
    {
        "mnist64"   : loadSklearnDigits8x8 ,
        "mnist784"  : loadSklearnDigits28x28 ,
        "cifar10"   : None,
    }

    DATASET_KEYWORD_TO_MODEL_BUILDER_CALLBACK = \
    {
        "mnist64"   : NeuralNetworks.NeuralNetworkModel.getConvNeuralNetworkA,
        "mnist784"  : NeuralNetworks.NeuralNetworkModel.getConvNeuralNetworkB,
        "cifar10"   : None,
    }

    def __init__(self,datasetKey):
        """ Constructor """
        self._datasetKey        = DatasetKey
        self._loaderCallback    = None
        self._modelCallback     = None
        
    def loadDatasetBunch(self):
        """ Load in the Dataset """
        if (self._loaderCallback is None):
            errMsg = "DatasetHandler.loadDataset() - Loader Callback is not set"
            raise RuntimeError(errMsg)
        return self._loaderCallback.__call__()
    
    def initModel(self):
        """ Return a constructed tf Neural Network that compliments the dataset """
        if (self._modelCallback is None):
            errMsg = "DatasetHandler.initModel() - Model Callback is not set"
            raise RuntimeError(errMsg)
        return self._modelCallback.__call__()

