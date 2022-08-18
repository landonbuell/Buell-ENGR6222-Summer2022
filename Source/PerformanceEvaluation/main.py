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

    # Path To Runs + Names
    runPaths = [
        "baseline_Mnist784_v0",
        "caseStudy1_Mnist784_v0",
        "caseStudy2_Mnist784_v0",
        "caseStudy3_Mnist784_v0",
        "caseStudy4_Mnist784_v0"
        ]

    runNames = [
        "Baseline",
        "Case-Study #1",
        "Case-Study #2",
        "Case-Study #3",
        "Case-Study #4"
        ]
    allRuns = [os.path.join(OUTPUT_PARENT,x) for x in runPaths]
    
    # Create the Classes to Evaluate + The Training History

    histories = Evaluators.TrainHistories(allRuns,runNames,EXPORT_PATH)
    histories.run()

    # Terminate
    sys.exit(0)