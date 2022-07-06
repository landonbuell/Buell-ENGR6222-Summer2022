"""
Author:         Landon Buell
Date:           July 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        Model
File:           Utilities.py
"""

        #### IMPORTS ####

from typing_extensions import runtime
import numpy as np

import PipelineTasks

        #### CLASS DEFINTIONS ####

class Pipeline:
    """ Pipeline is a list of tasks to be executed in order """

    def __init__(self,experiment):
        """ Constructor """
        self._queue         = np.array([],dtype=object)
        self._experiment    = experiment

    def __del__(self):
        """ Destructor """
        self._queue = None

    # Getters and Setters

    def getSize(self):
        """ Return the size of the pipeline """
        return self._queue.shape[0]

    def getExperiment(self):
        """ Get the Experiment that owns this queue """
        return self._experiment

    # Public Interface

    def registerTask(self,taskItem):
        """ Register and Item into the queue """
        if (isinstance(taskItem,PipelineTasks.PipelineTask) == False):
            # Item is NOT a pipeline Task
            msg = "Registered item must be a child class of PipelineTasks.PipelineTask. "
            msg += "Instead got {0}".format(taskItem.__class__)
            raise RuntimeError(msg)
        # Add the Task to the Queue and 
        self._queue = np.append(self._queue,taskItem)
        taskItem.setOwner(self)
        return self

    def evaluate(self,*args):
        """ Evaluate the Pipline line """
        self.evaluatePreprocessCallbacks()
        for item in self._queue:
            item.call()
        # After the main Pipeline
        self.evaluatePostprocessCallbacks()
        return self

    # Private Interface



    # Magic Methods

    def __len__(self):
        """ Get the size of current pipline """
        return self._queue.shape[0]

class PipelineTask:
    """ Abstract base class for a pipeline item """

    def __init__(self,name,callbacks=None,*args,**kwargs):
        """ Constructor """
        self._name          = name
        self._owner         = None
        self._callbacks     = []
        self._description   = ""
        # Check for callbacks to register
        if (callbacks is not None):
            self._callbacks = [x for x in callbacks]

    def __del__(self):
        """ Destructor """

    # Getters and Setters 

    def getName(self):
        """ Return the name of this task """
        return self._name

    def getOwner(self):
        """ Get the Pipeline that owns this task instance """
        return self._owner

    def setOwner(self,ownerPipeline):
        """ Set the Pipeline that owns this task """
        self._owner = ownerPipeline
        return self

    # Public Interface

    def describe(self):
        """ Return a description of this instance """
        print(self._description)

    def call(self,inputs=None):
        """ Invoke this task w/ Input Arguments """
        outputs = None
        return outputs

    def hasOwner(self):
        """ Return T/F if this task belongs to a pipeline """
        return (self._owner is not None)

    # Protected Interface

    def evaluateCallbacks(self):
        """ Evaluate Callbacks within this Task """
        for item in self._callbacks:
            item()
        return self
    
class LoadDataset(PipelineTask):
    """ Task to load in a specfied dataset """

    def __init__(self,datasetCode):
        """ Constructor """
        super().__init__("LoadDataset")
        self._datasetCode   = datasetCode
        self._numClasses    = 0
        self._loaderMethod  = self.

    def __del__(self):
        """ Destructor """

    # Public Interface

    def call(self,inputs):
        """ Load in this last + Return the Output """
        
    # Protected Interface


    # Static Interface

    @staticmethod
    def loadSklearnDigits8x8():
        """ Load 8x8 digits from Sklearn data set """
        return sklearn.datasets.load_digits(return_X_y=True)

    @staticmethod
    def loadSklearnDigits28x28():
        """ Load 28 x 28 digits from Sklearn Data set """
        return sklearn.datasets.fetch_openml("mnist_784")
