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

        #### FUNCTION DEFINTIONS ####

def plot2D(xData,yData,labels,title):
    """ Plot Something 2D """
    plt.figure(figsize=(16,12),tight_layout=True)
    #plt.title(title,size=40,weight='bold')
    plt.xlabel("Training Index",size=40,weight='bold')
    plt.ylabel(title,size=40,weight='bold')
    for i in range(len(labels)):
        plt.plot(xData[i],yData[i],label=labels[i])
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    return None


        #### CLASS DEFINITIONS ####

class RunEvaluator:
    """ Evaluate a Single Run Folder """

    def __init__(self,runPath,runName,expPath):
        """ Constructor """
        self._runPath = runPath
        self._runName = runName
        self._expPath = expPath

    def __del__(self):
        """ Destructor """
        pass

    # Public Interface

    def run(self):
        """ Run the Evaluator """
        self.plotLossHistory()
        return self


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

    def getAllTrainingHistoryPaths(self):
        """ Get all paths to Test Results in this folder """
        items = os.listdir(self._runPath)
        paths = []
        for x in items:
            if x.startswith("trainingHistory_"):
                y = os.path.join(self._runPath,x)
                paths.append(y)
        return paths

    def plotLossHistory(self):
        """ Plot the Training History for Each Run """
        historyFiles = self.getAllTrainingHistoryPaths()
        losses = [None] * len(historyFiles)
        epochs = [None] * len(historyFiles)
        labels = ["RUN {0}".format(x) for x in range(len(historyFiles))]
        # Collect all of the losses
        for i in range(len(historyFiles)):
            frame = pd.read_csv(historyFiles[i],index_col=0)
            losses[i] = frame["Loss"].to_numpy()
            epochs[i] = np.arange(len(losses[i]))
        # Plot all of them
        plot2D(epochs,losses,labels,"Loss History")

        return self

