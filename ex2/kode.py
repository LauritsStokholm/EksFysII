# Preamble
import pandas as pd

# For generating data
import os # Operative system commands
import csv  # reading csv

# Usual mathematics
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy import stats 


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
#data_path = os.path.join(os.getcwd(), 'Data') # ./Data
#data_files = os.listdir(data_path) # Files in ./Data
#data = data_files[2]
#data_dir = os.path.join(data_path, data)
#print(data)
#
#df = pd.read_csv(data_dir, skiprows=[1, 2])
#print(df.head())
#print(df.Time.head())
#print(df['ChannelA'])
#print(df.ChannelB.head())



# Filtering .csv files in ./Data
# Iterating index of data
#for ele in range(len(data_files)):
#    if '.csv' in data_files[ele]:
#        csv_path = os.path.join(data_path, data_files[ele])
#        with open(csv_path, 'r') as csv_file:
#            csv_file = csv.reader(csv_file)
#            for row in csv_file:
#                lists[ele].append(row)

#print(lists)

""" Experimental Setup 3: Mach Zehnder Interferometer """
# Importing data:
data_path = os.path.join(os.getcwd(), 'DataII')
data_files = os.listdir(data_path)

data_dir = os.path.join(data_path, data_files[1])
df = pd.read_csv(data_dir, skiprows=[1])
#print(df.head())
#print(df.Tid.head())
df.set_index('Tid', inplace=True)
df.plot()
plt.show()
