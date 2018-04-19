# # # # # # # # # # # # # # # Loading Libraries # # # # # # # # # # # # # # # # 

# Usual mathematics
import os
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy import stats 
from scipy import signal
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

# MatploLib running TeX
params = {'legend.fontsize'     : '20',
          'axes.labelsize'      : '20',
          'axes.titlesize'      : '20',
          'xtick.labelsize'     : '20',
          'ytick.labelsize'     : '20',
          'legend.numpoints'    : 1,
          'text.latex.preamble' : [r'\usepackage{siunitx}',
                                   r'\usepackage{amsmath}'],
          'axes.spines.right'   : False,
          'axes.spines.top'     : False,
          'figure.figsize'      : [8.5, 6.375],
          'legend.frameon'      : False
          }

plt.rcParams.update(params)
plt.rc('text',usetex =True)
plt.rc('font', **{'family' : "sans-serif"})



""" Experimental setup 2: Michelson Interferometer"""


# Importing Data

# Current working directory
data_path = os.path.join(os.getcwd(), 'DataII')

# List of files in ./Data
data_files = os.listdir(data_path)

# Picking out .csv files (datafiles)
data = [x for x in data_files if '.csv' in x]

# Paths for chosen datafiles
data_dir = [os.path.join(data_path, x) for x in data]

# Picking out a file to work on
data_file = data_dir[0]


