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

    def __init__(self):
        """ Constructor """
        self._queue         = np.array([],dtype=object)
        self._preCallbacks  = []
        self._postCallbacks = []

    def __del__(self):
        """ Destructor """
        self._queue = None

    # Getters and Setters

    def getSize(self):
        """ Return the size of the pipeline """
        return self._queue.shape[0]

    # Public Interface

    def registerTask(self,taskItem):
        """ Register and Item into the queue """
        if (isinstance(taskItem,PipelineTasks.PipelineTask) == False):
            # Item is NOT a pipeline Task
            msg = "Registered item must be a child class of PipelineTasks.PipelineTask. "
            msg += "Instead got {0}".format(taskItem.__class__)
            raise RuntimeError(msg)
        self._queue = np.append(self._queue,taskItem)
        return self

    def evaluate(self,*args):
        """ Evaluate the Pipline line """
        self.evaluatePreprocessCallbacks()
        for item in self._queue:
            item.call()
        # After the main Pipeline
        self.evaluatePostprocessCallbacks()
        return self

    def registerPreprocessCallbacks(self,callback):
        """ Register a method to be evaluated before the pipeline """
        self._preCallbacks.append(callback)
        return self

    def registerPostprocessCallbacks(self,callback):
        """ Register a method to be evaluated after the pipeline """
        self._postCallbacks.append(callback)
        return self

    # Private Interface

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

class PipleineInitializationStrategies:
    """ Static class of methods to initialize a pipeline """

    def __init__(self):
        """ Dummy Constructor """
        msg = "{0} : Is a static class. Cannot make instance".format(self.__class__)
        raise RuntimeError(msg)

    # Public Interface

    # TODO: Implement these for each step in the experiment