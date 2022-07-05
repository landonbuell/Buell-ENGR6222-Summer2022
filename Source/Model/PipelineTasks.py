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

        #### CLASS DEFINTIONS ####

class PipelineTask:
    """ Abstract base class for a pipeline item """

    def __init__(self,name,callbacks=None,*args,**kwargs):
        """ Constructor """
        self._name          = name
        self._callbacks     = []
        self._description   = None
        # Check for callbacks to register
        if (callbacks is not None):
            self._callbacks = [x for x in callbacks]

    def __del__(self):
        """ Destructor """

    # Getters and Setters 

    def getName(self):
        """ Return the name of this task """
        return self._name

    def getDescription(self):
        """ Return a description of this instance """
        return self._description

    # Public Interface


    def call(self,inputs=None):
        """ Invoke this task w/ Input Arguments """
        outputs = None
        return outputs

    # Protected Interface

    def evaluateCallbacks(self):
        """ Evaluate Callbacks within this Task """
        for item in self._callbacks:
            item()
        return self
    
class LoadDataset(PipelineTask):
    """ Task to load in a specfied dataset """

