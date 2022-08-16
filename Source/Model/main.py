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

    # Experiment Output Paths
    outputPaths = [
        #"baseline_Mnist784_v0",
        #"caseStudy1_Mnist784_v0",
        "caseStudy2_Mnist784_v0",
        "caseStudy3_Mnist784_v0",
        "caseStudy4_Mnist784_v0"]

    # Experiment Instances
    experimentCallbacks = [
        #PresetExperiments.getBaselineDigits28x28,
        #PresetExperiments.getCaseStudy1,
        PresetExperiments.getCaseStudy2,
        PresetExperiments.getCaseStudy3,
        PresetExperiments.getCaseStudy4 ]

    # For Each Experiment
    for outputPath,callback in zip(outputPaths,experimentCallbacks):
        app = callback.__call__(outputPath)

        # Run the Experiment + Export Configuration
        app.initialize()
        app.run()
        app.exportConfiguration()

    # Exit
    sys.exit()
