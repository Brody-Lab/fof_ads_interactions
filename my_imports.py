# GLOBAL IMPORTS
import os
import copy
import warnings
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
import run_params as SPEC


# figure plotting settings
# plt.style.use('seaborn-deep')
plt.rcParams['figure.dpi']= 150
plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.major.size'] = 2 
plt.rcParams['ytick.major.size'] = 2 
plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Spectral(np.linspace(0,1,8)))
plt.rcParams['figure.max_open_warning'] = 0

# confirm settings
print("\n\nImported settings! Check that the paths are correct (specified in run_params.py):\n")
print("BASEPATH = {}".format(SPEC.BASEPATH))
print("DATA DIR = {}".format(SPEC.DATADIR))
print("FIGURE DIR = {}".format(SPEC.FIGUREDIR))
print("RESULT DIR = {}".format(SPEC.RESULTDIR))



