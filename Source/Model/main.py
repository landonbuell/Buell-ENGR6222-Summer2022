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
import sys

import Experiments

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Setup
    EXIT_STATUS = 0

    # Experiment
    outputPath = "..\\..\\outputs\\baselineMnist784_v0"
    #app = Experiments.Experiment.getBaselineDigits8x8(outputPath)
    app = Experiments.Experiment.getBaselineDigits28x28(outputPath)
    
    # Run the Experiment + Export Configuration
    app.initialize()
    app.run()
    app.exportConfiguration()

    # Exit
    sys.exit()
