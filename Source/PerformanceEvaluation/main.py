"""
Author:         Landon Buell
Date:           August 2022
Repo:           Buell-ENGR6222-Summer2022
Solution:       Source
Project:        PeformanceEvaluation
File:           main.py
"""

        #### IMPORTS ####

import os
import sys

import Evaluators

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Set some Variables
    OUTPUT_PARENT = "..\\..\\outputs"
    EXPORT_PATH = "..\\..\\analysis"
    runPaths = ["baseline_Mnist784_v0"]
    runNames = ["Baseline"]
    allRuns = [os.path.join(OUTPUT_PARENT,x) for x in runPaths]

    # Create the Classes to Evaluate
    for (run,name) in zip(allRuns,runNames):
        app = Evaluators.RunEvaluator(run,name,EXPORT_PATH)
        app.run()


    



    # Terminate

    sys.exit(0)