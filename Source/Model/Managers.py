"""
Author:         Landon Buell
Date:           July 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        Model
File:           Utilities.py
"""

        #### IMPORTS ####

import os

import numpy as np
import pandas as pd

import sklearn.datasets
from sklearn.model_selection import train_test_split


import NeuralNetworks

        #### FUNCTION DEFINITIONS ####


def loadSklearnDigits8x8(datasetManager):
    """ Load 8x8 digits from Sklearn data set """
    datasetBunch = sklearn.datasets.load_digits()
    datasetManager.registerDatasetBunch( datasetBunch )
    return None

def loadSklearnDigits28x28(datasetManager):
    """ Load 28 x 28 digits from Sklearn Data set """
    datasetBunch = sklearn.datasets.fetch_openml("mnist_784")
    datasetManager.registerDatasetBunch( datasetBunch )
    return None

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

        #### CLASS DEFINTIONS ####

class AbstractManager:
    """ Parent Class for All Managers """

    def __init__(self,name):
        """ Constructor """
        self._name      = name
        self._owner     = None

    def __del__(self):
        """ Destructor """
        self._name      = None
        self._owner     = None

    # Getters and Setters

    def getName(self):
        """ Return Name """
        return self._name

    def getOwner(self):
        """ Get the Experiment that Owns this Manager """
        return self._owner

    def setOwner(self,parentExperiment):
        """ Set the Experiment that owns this manager """
        self._owner = parentExperiment
        return self

class DatasetManager(AbstractManager):
    """ Dataset Manager loads + preprocess + splits a data set """

    def __init__(self,datasetkeyWord):
        """ Constructor """
        super().__init__("DatasetManager")
        self._datasetCode       = datasetkeyWord
        self._loaderCallback    = None

        self._designMatrix  = None
        self._targetLabels  = None

        self._targetNames       = []
        self._description       = ""

        self._inputShape        = (0,)
        self._numClasses        = 0
        self._numSamples        = 0

              
    def __del__(self):
        """ Destructor """
        super().__del__()

    # Getters and Setters

    def getDatasetKeyWord(self):
        """ Return the Keyword for this dataset """
        return self._datasetCode

    def getDatasetCallbacks(self):
        """ Return the Callback that loads the dataset """
        return self._loaderCallback

    def getDesignMatrix(self):
        """ Return the Design Matrix """
        return self._designMatrix

    def getTargetLabels(self):
        """ Return the Target Labels """
        return self._targetLabels

    def getTargetNames(self):
        """ Return list of target Names """
        return self._targetNames

    def getDataDescription(self):
        """ Return the dataset description """
        return self._description

    def getInputShape(self):
        """ Return the shape of each samples """
        return self._inputShape

    def getNumClasses(self):
        """ Return the number of unqiue classes in the dataset """
        return self._numClasses

    def getNumSamples(self):
        """ Return the number of samples in the dataset """
        return self._numSamples


    # Public Interface

    def registerDatasetBunch(self,bunch):
        """ Register a Dataset to this instance """
        self._designMatrix  = bunch.images
        self._targetLabels  = bunch.target

        self._targetNames   = bunch.target_names
        self._description   = bunch.DESCR

        self._inputShape    = bunch.images.shape[1:]
        self._numClasses    = bunch.target_names.shape[0]

        self._numSamples    = self._designMatrix.shape[0]

        if (len(self._inputShape) < 3):
            oldShape = self._inputShape
            newShape = tuple([self._numSamples] + [x for x in oldShape] + [1])
            self._inputShape = newShape[1:]
            self._designMatrix = self._designMatrix.reshape(newShape)
       
        return self

    def loadDataset(self):
        """ Load in this last + Return the Output """
        self.registerCallbackFromDatasetCode()
        if (self._loaderCallback is None):
            msg = "Dataset loading callback not set!"
            raise RuntimeError(msg)

        # Invoke the callback to get the dataset + parse the bunch
        self._loaderCallback.__call__(self)
           
        return self
  
    @staticmethod
    def oneHotEncode(targetVector,numClasses):
        """ One - Hot encode samples for multi classification """
        oneHot = np.zeros(shape=(targetVector.shape[0],numClasses),dtype=np.int16)
        for i,y in enumerate(targetVector):
            oneHot[i,y] = 1
        return oneHot

    # Protected Interface

    def registerCallbackFromDatasetCode(self):
        """ Choose Callback based on Dataset key """
        successful = False
        codeWord = self._datasetCode
        try:
            self._loaderCallback = DATASET_KEYWORD_TO_LOADER_CALLBACK[codeWord]
            successful = True
        except KeyError as err:
            msg = "Dataset Keyword {0} was not recognized.".format(codeWord)
            raise err(msg)
        return successful

class ConvNeuralNetworkBuilder(AbstractManager):
    """ Class to build + Return a TF Convolutional Neural network """

    def __init__(self,datasetCode):
        """ Constructor """
        super().__init__("ConvNeuralNetworkBuilder")
        self._datasetCode       = datasetCode
        self._modelCallback     = None
        self._randomSeed        = np.random.randint(0,1e6,1)
        self._model             = None

    def __del__(self):
        """ Destructor """

    # Getters and Setters

    def getInputShape(self):
        """ Return the input shape from the dataset manager """
        return self.getOwner().getDatasetManager().getInputShape()

    def getNumClasses(self):
        """ Return the number of classes from the dataset manager """
        return self.getOwner().getDatasetManager().getNumClasses()

    def getModel(self):
        """ Return the stored TF Model """
        return self._model

    def getSeed(self):
        """ Get the current random seed """
        return self._randomSeed

    def setSeed(self,seed):
        """ Set the current random seed """
        self._randomSeed = seed
        return self

    # Public Interface 

    def buildModel(self):
        """ Call this Instance """
        self.registerCallbackFromDatasetCode()
        if (self._modelCallback is None):
            msg = "Dataset loading callback not set!"
            raise RuntimeError(msg)

        model = self._modelCallback.__call__(
            self.getInputShape(),
            self.getNumClasses(),
            self.getSeed())

        # Show Summary of Model
        print(model.summary())
        self._model = model

        return self

    # Protected Interface

    def registerCallbackFromDatasetCode(self):
        """ Choose Callback based on Dataset key """
        successful = False
        codeWord = self._datasetCode
        try:
            self._modelCallback = DATASET_KEYWORD_TO_MODEL_BUILDER_CALLBACK[codeWord]
            successful = True
        except KeyError as err:
            msg = "Dataset Keyword {0} was not recognized.".format(codeWord)
            raise err(msg)
        return successful

class TrainingManager(AbstractManager):
    """ Class to Manage Model Training """

    def __init__(self,trainSize,trainEpochs):
        """ Constructor """
        super().__init__("TrainingManager")
        self._trainSize             = trainSize
        self._trainEpochs           = trainEpochs
        self._batchSize             = 32
    
    def __del__(self):
        """ Destructor """

    # Getters and Setters

    def getTrainSize(self):
        """ Get percentage of data used for training """
        return self._trainSize

    def getTestSize(self):
        """ Get percentage of data used for testing """
        return (1.0 - self._trainSize)

    def getNumEpochs(self):
        """ Get the number of epochs in training process """
        return self._trainEpochs

    # Public Interface

    def splitTrainTest(self):
        """ Perform Train-Test Split """
        return train_test_split(
            self.getOwner().getDatasetManager().getDesignMatrix(),
            self.getOwner().getDatasetManager().getTargetLabels(),
            train_size=self._trainSize)

    def trainModel(self,X,y,iterNum):
        """ Train a Model w/ X + y Data """
        model = self.getOwner().getModelManager().getModel()
        rundataMgr = self.getOwner().getRundataManager()

        numClasses = self.getOwner().getDatasetManager().getNumClasses()
        y = DatasetManager.oneHotEncode(y,numClasses)

        # Train + Get History
        history = model.fit(X,y,batch_size=self._batchSize,epochs=self._trainEpochs)
        rundataMgr.updateTrainingHistory(history)
        rundataMgr.exportTrainingHistory("trainingHistory_{0}.csv".format(iterNum))

        return self

    def testModel(self,X,y,iterNum):
        """ Test a Model w/ X + y Data """
        model = self.getOwner().getModelManager().getModel()
        rundataMgr = self.getOwner().getRundataManager()

        # Test + Get Predictions
        outputs = model.predict(X)
        rundataMgr.updateTestResults(outputs,y)
        rundataMgr.exportTestResults("testResults_{0}.csv".format(iterNum))

        return self

class ExportManager(AbstractManager):
    """ Class to Export the results of an Experiment """

    def __init__(self,outputPath):
        """ Constructor """
        super().__init__("ExportManager")
        self._outputPath        = os.path.abspath(outputPath)
        self._trainHistory      = NeuralNetworks.RunningHistory()
        self._testResults       = NeuralNetworks.TestResults()

        self.makeOutputPath()

    def __del__(self):
        """ Destructor """

    # Getters and Setters

    def getOutputPath(self):
        """ Return the output path """
        return self._outputPath

    # Public Interface 

    def updateTrainingHistory(self,history):
        """ Update this history w/ training data """
        self._trainHistory.update(history)
        return self

    def updateTestResults(self,results,labels):
        """ Update this results w/ testing data """
        self._testResults.update(results,labels)
        return self

    def exportTrainingHistory(self,fileName):
        """ Export history to CSV file """
        outputFrame = pd.DataFrame(data=self._trainHistory.getMetricsDictionary())
        outputPath = os.path.join(self._outputPath,fileName)
        outputFrame.to_csv(outputPath,sep=",",header=True,index=True)
        return self

    def exportTestResults(self,fileName):
        """ Export results to CSV file """
        outputFrame = pd.DataFrame(data=self._testResults.getResultsDictionary())
        outputPath = os.path.join(self._outputPath,fileName)
        outputFrame.to_csv(outputPath,sep=",",header=True,index=True)
        return self

    def clearTrainingHistory(self):
        """ Clear the history instance """
        self._trainHistory.clear()
        return self

    def clearTestingResults(self):
        """ Clear the results instance """
        self._testResults.clear()
        return self

    # Private Interface

    def makeOutputPath(self):
        """ Make the output directory """
        if (os.path.isdir(self._outputPath) == True):
            # Output path Exists
            msg = "WARNING: Output path '{0}' already exists. Content may be overwritten".format(
                self._outputPath)
            print(msg)
        else:
            # Output path does not exist
            msg = "Sending outputs to path: '{0}' ".format(self._outputPath)
            print(msg)
            os.makedirs(self._outputPath)
        return self
        


