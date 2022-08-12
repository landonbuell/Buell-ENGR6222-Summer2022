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

import PresetExperiments

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Setup
    EXIT_STATUS = 0

    # Experiment
    outputPath = "caseStudy1_Mnist784_v0"
    #app = PresetExperiments.getBaselineDigits28x28(outputPath)
    app = PresetExperiments.getCaseStudy1(outputPath)
    #app = PresetExperiments.getCaseStudy2(outputPath)
    #app = PresetExperiments.getCaseStudy3(outputPath)
    #app = PresetExperiments.getCaseStudy4(outputPath)

    # Run the Experiment + Export Configuration
    app.initialize()
    app.run()
    app.exportConfiguration()

    # Exit
    sys.exit()
