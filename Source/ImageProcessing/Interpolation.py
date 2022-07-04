"""
Author:         Landon Buell
Date:           July 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        ImageProcessing
File:           Interpolation.py
"""

        #### IMPORTS ####

import numpy as np
import scipy.interpolate as interpolate

        #### CLASS DEFINITIONS ####

class ScipyInterpolators:
    """ Static Class of methods for getting scipy 2D Interpolators """

    def __init__(self):
        """ Constructor """
        msg = "{0} : Is a static class. Cannot make instance".format(self.__class__)
        raise RuntimeError(msg)

    # Public Interface

    

class Interpolator2D:
    """ Abstract Base class for all 2D interpolation classess """

    def __init__(self,name):
        """ Constructor """
        self._name = name

    def __del__(self):
        """ Destructor """

    # Getters and Setters

    # Public Interface

    def call(self,image):
        """ Invoke the instance on an image """
        return image

    # Protected Interface

    def isNdim(self,item,ndim=2):
        """ Return T/F if object is of the specified dimension """
        if (isinstance(item,np.ndarray) == False):
            msg = "{0} : isNdim() - input item must be type np.ndarray".format(self.__class__)
            raise RuntimeError(msg)
        return (item.ndim == ndim)

    # Magic Methods

    def __call__(self,image):
        """ Convenience function for invoking instance """
        return self.call(image)

    def __repr__(self):
        """ Debug Representation of Instance """
        return "{0}{1}".format(self.__class__,hex(id(self)))

