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
import sklearn.metrics

        #### GLOBAL CONSTANTS ####

EPSILON = np.array([1e-3],dtype=np.float64)

NUM_TRAINING_ITERS = 1750
NUM_EXPERIMENT_REPS = 10
NUM_CLASSES = 10
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

def plotBarChart(yData,labels,title,yticks=None):
    """ Make a Bar Chart """
    plt.figure(figsize=(12,8),tight_layout=True)
    plt.ylabel(title,size=30,weight='bold')

    xData = np.arange(0,len(yData),1,dtype=np.int16)
    widths = np.ones(len(yData)) * 0.8
    colors = ['red','lightblue','green','purple','darkorange']
    plt.bar(xData,yData,width=widths,color=colors)

    plt.xticks(xData,labels=labels,size=25,rotation=20)
    if (yticks is not None):
        plt.yticks(yticks)
        plt.ylim(min(yticks),max(yticks))

    # Annonate
    for i in range(len(yData)):
        txt = str(round(yData[i],4))
        plt.annotate(txt,(xData[i] - 0.25 ,0.15),color='black',weight='bold',size=20)
    
    plt.grid()
    plt.show()
    plt.close()
    return None

def categoricalCrossentropy(x,y):
    """ Compute + Return Categorical cross entropy objective """
    cxe = np.empty(shape=(x.shape[0]),dtype=np.float64)
    for i in range(cxe.shape[0]):
        cxe[i] = np.dot(x[i],np.log(y[i] + EPSILON))
    return np.mean(cxe,axis=0)


#### CLASS DEFINITIONS FOR TRAINING HISTORY ####

class MetricHistories:
    """ Structure to hold Metrics """

    def __init__(self,numFiles,numIters=NUM_TRAINING_ITERS):
        """ Constructor """
        self.losses     = np.empty(shape=(numFiles,numIters))
        self.precisions = np.empty(shape=(numFiles,numIters))
        self.recalls    = np.empty(shape=(numFiles,numIters))
        self.f1Scores   = np.empty(shape=(numFiles,numIters))
        self.epochs     = np.arange(numIters,dtype=np.int16)

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
        metricStruct = MetricHistories(len(historyFiles))
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

    def getAllTrainingHistoryPaths(self):
        """ Get all paths to Test Results in this folder """
        items = os.listdir(self._runPath)
        paths = []
        for x in items:
            if x.startswith("trainingHistory_"):
                y = os.path.join(self._runPath,x)
                paths.append(y)
        return paths

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

#### CLASS DEFINITIONS FOR TESTINBG RESULTS ####

class MetricResults:
    """ Structure to hold Metrics """

    def __init__(self,numFiles):
        """ Constructor """
        self.losses     = np.empty(shape=(numFiles,))
        self.precisions = np.empty(shape=(numFiles,))
        self.recalls    = np.empty(shape=(numFiles,))
        self.f1Scores   = np.empty(shape=(numFiles,))
        self.epochs     = np.arange(0,1,dtype=np.int16)

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

class ModelTestResult:
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
        resultFiles = self.getAllTestResultPaths() 
        metricStruct = MetricResults(len(resultFiles))
        # Iterate through all Files
        for i in range(len(resultFiles)):
            frame = pd.read_csv(resultFiles[i],index_col=0)
            truth = frame['Labels'].to_numpy()
            guess = frame['Predicitions'].to_numpy()

            # One-Hot
            #oneHotTruth = ModelTestResult.oneHotEncode(truth,NUM_CLASSES)
            #rawOutput = ModelTestResult.organizeRawOutput(frame)

            # Extract into metrics
            #metricStruct.losses[i]      = categoricalCrossentropy(oneHotTruth,rawOutput)
            metricStruct.precisions[i]  = sklearn.metrics.precision_score(truth,guess,average="macro")
            metricStruct.recalls[i]     = sklearn.metrics.recall_score(truth,guess,average="macro")
            metricStruct.f1Scores[i]    = sklearn.metrics.f1_score(truth,guess,average="macro")

        # Return the populated struct
        return metricStruct

    # Private Interface

    def getAllTestResultPaths(self):
        """ Get all paths to Test Results in this folder """
        items = os.listdir(self._runPath)
        paths = []
        for x in items:
            if x.startswith("testResults_"):
                y = os.path.join(self._runPath,x)
                paths.append(y)
        return paths

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

    # Static Interface

    @staticmethod
    def oneHotEncode(targetVector,numClasses):
        """ One - Hot encode samples for multi classification """
        oneHot = np.zeros(shape=(targetVector.shape[0],numClasses),dtype=np.int16)
        for i,y in enumerate(targetVector):
            oneHot[i,y] = 1
        return oneHot

    @staticmethod
    def organizeRawOutput(dataFrame):
        """ Organize the Raw output of the model into a single array """
        numSamples = len(dataFrame)
        outputs = np.empty(shape=(numSamples,NUM_CLASSES))
        for i in range(NUM_CLASSES):
            key = "class_{0}".format(i)
            outputs[:,i] = dataFrame[key].to_numpy()
        return outputs

class TestResults:
    """ Evaluate Multiple Model Test Results """

    def __init__(self,runPaths,runNames,exptPath):
        """ Constructor """
        self._runPaths = runPaths
        self._runNames = runNames
        self._exptPath = exptPath

    def __del__(self):
        """ Destructor """
        pass

    def run(self):
        """ Run the Evaluator """
        metricStructs = []
        for (run,name) in zip(self._runPaths,self._runNames):
            results = ModelTestResult(run,name,self._exptPath)
            metrics = results.run()
            metricStructs.append(metrics)

        # Plot everything!
        #self.plotLossScore(metricStructs)
        self.plotPrecisionScore(metricStructs)
        self.plotRecallScore(metricStructs)
        self.plotF1Score(metricStructs)

        # Return!
        return self

    def plotLossScore(self,metricStructs):
        """ Plot All Avg Loss Scores """
        losses = [x.getAverageLoss() for x in metricStructs]
        labels = self._runNames
        yTicks = np.arange(0,11,1)
        plotBarChart(losses,labels,"Average Objective",yTicks)
        return self
    
    def plotPrecisionScore(self,metricStructs):
        """ Plot All Avg Loss Scores """
        precisions = [x.getAveragePrecisions() for x in metricStructs]
        labels = self._runNames
        yTicks = np.arange(0,1.1,0.1)
        plotBarChart(precisions,labels,"Average Precision",yTicks)
        return self

    def plotRecallScore(self,metricStructs):
        """ Plot All Avg Loss Scores """
        recalls = [x.getAverageRecall() for x in metricStructs]
        labels = self._runNames
        yTicks = np.arange(0,1.1,0.1)
        plotBarChart(recalls,labels,"Average Recall",yTicks)
        return self

    def plotF1Score(self,metricStructs):
        """ Plot All Avg Loss Scores """
        f1Scores = [x.getAverageF1Score() for x in metricStructs]
        labels = self._runNames
        yTicks = np.arange(0,1.1,0.1)
        plotBarChart(f1Scores,labels,"Average F1-Score",yTicks)
        return self



