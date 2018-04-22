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



""" Experimental setup 3: Mach Zehnder Interferometer"""


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
data_file = data_dir[9]


# Data Analysis
''' This is the data analysis per datafile, defined for determining the
material constant'''

# Reading csv
df = pd.read_csv(data_file, skiprows=[0, 1, 2], names=['Time', 'ChannelA'])

# Slicing data to seperate air flow in/out
df['Data1'] = df.ChannelA.where(df.ChannelA.index < df.ChannelA.size/2)
df['Data2'] = df.ChannelA.where(df.ChannelA.index > df.ChannelA.size/2)

#max_idx = df.Data1.idxmax()
#min_idx = df.Data1.idxmin()
#
#interval = abs(df.Data1.idxmax() - df.Data1.idxmin())
##order_vals = interval/25
#order_vals = 1000
#print(order_vals)
#
#maximas1 = sp.signal.argrelmax(df.Data1.as_matrix(), order=order_vals)
#maximas2 = sp.signal.argrelmax(df.Data2.as_matrix(), order=order_vals)

#maximas1 = maximas1[0]
#maximas2 = maximas2[0]
#print(maximas1)
#print(maximas2)
#
#xvals1 = df.Data1.loc(maximas1)

plt.figure()
#plt.plot(df.Data1.index())
df.Data1.plot()
df.Data2.plot()


plt.show()


m = 22


