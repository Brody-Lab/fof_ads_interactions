import os
import platform

# SPECIFY OTHER GLOBAL SPECS
RATS = ["X046", "X062", "X087", "A294", "A297"]
REGIONS = ["FOF", "ADS"]

# Sets paths based on detected os, linux or macOS
# Modify paths relevant to your OS:
BASEPATH = "" # to base directory
FIGUREDIR =  BASEPATH + "/figure_pdfs/" # directory to save figures
RESULTDIR =  BASEPATH + "/Code/saved_results/" # directory to save intermediate results
CODEDIR = BASEPATH + "/Code/" # directory with Code
DATADIR = BASEPATH + "Cells/"  # directory with raw ephys data 

# colors for plotting
COLS = {"FOF": [i/255 for i in [0, 164, 204]], 
        "ADS": [i/255 for i in [233,115,141]]}

