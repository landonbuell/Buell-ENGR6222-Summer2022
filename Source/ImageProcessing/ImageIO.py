"""
Author:         Landon Buell
Date:           July 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        ImageProcessing
File:           ImageIO.py
"""

        #### IMPORTS ####

import numpy as np
import sklearn.datasets

        #### CLASS DEFINITIONS ####

class ExistingDatasets:
    """ Static Class for loading in exisiting datasets """

    def __init__(self):
        """ Constructor """
        msg = "{0} : Is a static class. Cannot make instance".format(self.__class__)
        raise RuntimeError(msg)

    # Public Interface

    @staticmethod
    def oneHotEncode(lables,numClasses):
        """ One-Hot-Encode the output """
        y = np.zeros(shape=(len(labels),numClasses),dtype=np.int32)
        for i in lables:
            y[i] = 1
        return y

    @staticmethod
    def getDigitsSklearn():
        """ Get MNIST Digits Dataset as (x,y) """
        data = sklearn.datasets.load_digits(return_X_y=True)
        return data
