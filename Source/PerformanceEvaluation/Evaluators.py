"""
Author:         Landon Buell
Date:           August 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        PeformanceEvaluation
File:           Evaluators.py
"""

        #### IMPORTS ####

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

        #### GLOBAL CONSTANTS ####

NUM_TRAINING_ITERS = 1750
NUM_EXPERIMENT_REPS = 10
NUM_PIXELS = [729, 196, 676, 169]

        #### FUNCTION DEFINTIONS ####

def plot2D(xData,yData,labels,title,yticks=None):
    """ Plot Something 2D """
    plt.figure(figsize=(12,8),tight_layout=True)
    #plt.title(title,size=40,weight='bold')
    plt.xlabel("Training Index",size=30,weight='bold')
    plt.ylabel(title,size=30,weight='bold')
    for i in range(len(labels)):
        plt.plot(xData[i],yData[i],label=labels[i],
                 linestyle="--",linewidth=4)
    plt.legend(fontsize=25)
    if (yticks is not None):
        plt.yticks(yticks)
        plt.ylim(min(yticks),max(yticks))
    plt.grid()
    plt.show()
    plt.close()
    return None


        #### CLASS DEFINITIONS ####

class TrainHistories:
    """ Evaluate Multiple Model Training Histories """

    def __init__(self,runPaths,runNames,output):
        """ Constructors """
        self._runPaths = runPaths
        self._runNames = runNames
        self._exptPath = output

    def __del__(self):
        """ Destructor """
        pass

    def run(self):
        """ Run the Evaluator """
        metricStructs = []
        for (run,name) in zip(self._runPaths,self._runNames):
            history = ModelTrainHistory(run,name,self._exptPath)
            metrics = history.run()
            metricStructs.append(metrics)

        # Plot everything!
        self.plotLossHistory(metricStructs)
        self.plotPrecisionHistory(metricStructs)
        self.plotRecallHistory(metricStructs)
        self.plotF1ScoreHistory(metricStructs)

        # Return!
        return self

    # Private Interface

    def plotLossHistory(self,metricStructs):
        """ Plot All Avg Loss histories """
        losses = [x.getAverageLoss() for x in metricStructs]
        epochs = [x.epochs for x in metricStructs]
        labels = self._runNames
        yTicks = np.arange(0,11,1)
        plot2D(epochs,losses,labels,"Average Objective",yTicks)
        return self
    
    def plotPrecisionHistory(self,metricStructs):
        """ Plot All Avg Loss histories """
        precisions = [x.getAveragePrecisions() for x in metricStructs]
        epochs = [x.epochs for x in metricStructs]
        labels = self._runNames
        yTicks = np.arange(0,1.1,0.1)
        plot2D(epochs,precisions,labels,"Average Precision",yTicks)
        return self

    def plotRecallHistory(self,metricStructs):
        """ Plot All Avg Loss histories """
        recalls = [x.getAverageRecall() for x in metricStructs]
        epochs = [x.epochs for x in metricStructs]
        labels = self._runNames
        yTicks = np.arange(0,1.1,0.1)
        plot2D(epochs,recalls,labels,"Average Recall",yTicks)
        return self

    def plotF1ScoreHistory(self,metricStructs):
        """ Plot All Avg Loss histories """
        f1Scores = [x.getAverageF1Score() for x in metricStructs]
        epochs = [x.epochs for x in metricStructs]
        labels = self._runNames
        yTicks = np.arange(0,1.1,0.1)
        plot2D(epochs,f1Scores,labels,"Average F1-Score",yTicks)
        return self

class ModelTrainHistory:
    """ Evaluate a Single Run Folder """

    def __init__(self,runPath,runName,expPath):
        """ Constructor """
        self._runPath = runPath
        self._runName = runName
        self._expPath = expPath
        # Metrics
        self._epochs        = np.arange(NUM_TRAINING_ITERS)

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def run(self):
        """ Run the Evaluator """
        historyFiles = self.getAllTrainingHistoryPaths() 
        metricStruct = MetricHistory(len(historyFiles))
        # Iterate through all Files
        for i in range(len(historyFiles)):
            frame = pd.read_csv(historyFiles[i],index_col=0)
            metricStruct.losses[i]     = frame["Loss"].to_numpy()
            metricStruct.precisions[i] = frame["Precision"].to_numpy()
            metricStruct.recalls[i]    = frame["Recall"].to_numpy()
            metricStruct.f1Scores[i]   = frame["F1-Score"].to_numpy()

        # Plot Each Runs
        #self.plotLosses(metricStruct)
        #self.plotPrecisions(metricStruct)
        #self.plotRecalls(metricStruct)
        #self.plotF1Scores(metricStruct)

        # Return the populated struct
        return metricStruct

    # Private Interface

    def plotLosses(self,metricStruct):
        """ Plot Results of Runs """
        x = [metricStruct.epochs] * NUM_EXPERIMENT_REPS
        y = metricStruct.losses
        labels = ["RUN {0}".format(i) for i in range(NUM_EXPERIMENT_REPS)]
        plot2D(x,y,labels,"Losses")
        return None

    def plotPrecisions(self,metricStruct):
        """ Plot Results of Runs """
        x = [metricStruct.epochs] * NUM_EXPERIMENT_REPS
        y = metricStruct.precisions
        labels = ["RUN {0}".format(i) for i in range(NUM_EXPERIMENT_REPS)]
        plot2D(x,y,labels,"Precisions")
        return None

    def plotRecalls(self,metricStruct):
        """ Plot Results of Runs """
        x = [metricStruct.epochs] * NUM_EXPERIMENT_REPS
        y = metricStruct.recalls
        labels = ["RUN {0}".format(i) for i in range(NUM_EXPERIMENT_REPS)]
        plot2D(x,y,labels,"Recalls")
        return None

    def plotF1Scores(self,metricStruct):
        """ Plot Results of Runs """
        x = [metricStruct.epochs] * NUM_EXPERIMENT_REPS
        y = metricStruct.f1Scores
        labels = ["RUN {0}".format(i) for i in range(NUM_EXPERIMENT_REPS)]
        plot2D(x,y,labels,"F1-Scores")
        return None

    def getAllTestResultPaths(self):
        """ Get all paths to Test Results in this folder """
        items = os.listdir(self._runPath)
        paths = []
        for x in items:
            if x.startswith("testResults_"):
                y = os.path.join(self._runPath,x)
                paths.append(y)
        return paths

    def getAllTrainingHistoryPaths(self):
        """ Get all paths to Test Results in this folder """
        items = os.listdir(self._runPath)
        paths = []
        for x in items:
            if x.startswith("trainingHistory_"):
                y = os.path.join(self._runPath,x)
                paths.append(y)
        return paths

class MetricHistory:
    """ Structure to hold Metrics Histories """

    def __init__(self,numFiles):
        """ Constructor """
        self.losses     = np.empty(shape=(numFiles,NUM_TRAINING_ITERS))
        self.precisions = np.empty(shape=(numFiles,NUM_TRAINING_ITERS))
        self.recalls    = np.empty(shape=(numFiles,NUM_TRAINING_ITERS))
        self.f1Scores   = np.empty(shape=(numFiles,NUM_TRAINING_ITERS))
        self.epochs     = np.arange(NUM_TRAINING_ITERS,dtype=np.int16)

    def __del__(self):
        """ Destructor """
        pass

    def getAverageLoss(self):
        """ Return Average Loss """
        return np.mean(self.losses,axis=0)

    def getAveragePrecisions(self):
        """ Return Average Loss """
        return np.mean(self.precisions,axis=0)

    def getAverageRecall(self):
        """ Return Average Loss """
        return np.mean(self.recalls,axis=0)

    def getAverageF1Score(self):
        """ Return Average Loss """
        return np.mean(self.f1Scores,axis=0)
