# Loading Libraries

# Usual mathematics
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
from scipy import optimize as opt
from scipy import stats
from scipy import signal
#from scipy.signal import argrelextrema # local extremas

# Styling
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


# # # # # # # # # # # # # # # # Importing Data # # # # # # # # # # # # # # # # 

# Current working directory
data_path = os.path.join(os.getcwd(), 'Data')

# List of files in ./Data
data_files = os.listdir(data_path)

# Picking out .csv files (datafiles)
data = [x for x in data_files if '.csv' in x]

# Paths for chosen datafiles
data_dir = [os.path.join(data_path, x) for x in data]
data_dir = data_dir[9]
print(data_dir)

# # # # # # # # # # # # # # # # Define Analysis # # # # # # # # # # # # # # # # 

# Reading csv
df = pd.read_csv(data_dir, skiprows=[0, 1, 2], names=['Time',
                                                      'ChannelA',
                                                      'ChannelB'])

# Unit Conversion
df.ChannelA = df.ChannelA * 10**-3
df.Time = df.Time * 10**-3


# Indices of extrema
# Array of indices where data is minima / maxima
min_index = np.array(df.ChannelB.index.where(df.ChannelB ==
    df.ChannelB.min()).dropna())
max_index = np.array(df.ChannelB.index.where(df.ChannelB ==
    df.ChannelB.max()).dropna())



# Slicing (equal sizes)
if np.size(min_index) > np.size(max_index):
    min_index = min_index[:np.size(max_index)]
elif np.size(min_index) < np.size(max_index):
    max_index = max_index[:np.size(min_index)]
else: pass


# Converting (floats) to ints
min_index = min_index.astype(int)
max_index = max_index.astype(int)
index = np.vstack((min_index, max_index))
print(index)

print(min_index)
print(max_index)

print(min(max_index - min_index))
print(index[0][0])
print(index[1][0])

df.ChannelB.plot()
df.ChannelB[min(min_index[0], max_index[0]) :
            max(min_index[0], max_index[0])].plot()
df.ChannelB[min(index[1][0], index[1][0]): max(index[1][0], index[1][1])]
df.ChannelB[min(index[1][1], index[1][1]): max(index[1][1], index[1][1])]
df.ChannelB[min(index[1][2], index[1][2]): max(index[1][2], index[1][2])]
plt.show()

#max_index = df.ChannelB[df.ChannelB == df.ChannelB.max()]
#print(df.ChannelB[min_index])
#max_index = df.ChannelB.idxmax()
#print('The extremas are from ' + str(min_index) + ' to ' + str(max_index))

#df.ChannelB[:min_index].plot()

# Filtering data to first interval
#v_piezo   = df.ChannelB[min_index : max_index+1]
#intensity = df.ChannelA[min_index : max_index+1]
#print(intensity.describe())
#
## Smoothening intensity data by method
#df['intensity_hat'] = pd.Series()
#print(df.intensity_hat.describe())
##df['intensity_hat'][min_index:max_index+1] =\
##pd.Series(sp.signal.savgol_filter(intensity, 51, 3))
#print(df.intensity_hat.describe())
##intensity_hat = df.intensity_hat[min_index : max_index+1]
#df.fillna(0, inplace=True)
#
## Visualisation (+ smoothening by method)
#plt.figure()
#plt.title('Data')
#plt.xlabel('V piezo')
#plt.ylabel('Intensity')
#plt.plot(v_piezo, intensity)
#plt.plot(v_piezo, intensity_hat, label='Savitzky Golay smoothening')
#plt.grid()
#plt.legend(loc=3)

# Wavelenght of V
# Read off indices from plots (remember to corregate min_index)
# New method (yaay!)
#maximas = sp.signal.argrelextrema(np.array(intensity_hat),
#                                  np.greater,
#                                  order=500)
#max_index1_hat = maximas[0][0]
#max_index2_hat = maximas[0][1]
#
## Old method (buuh!)
##x12 = 13500 - min_index
##x21 = 15000 - min_index
##x22 = 20000 - min_index
##
### Finding extremas
##max_index1_hat = intensity_hat[:x12].idxmax(axis=1)
##max_index2_hat = intensity_hat[x21:x22].idxmax(axis=1)
#
#print('The extremas arrre '+str(max_index1_hat)+' and '+str(max_index2_hat))
#
### Visualisation of chosen indices
#plt.figure()
#plt.title('Data')
#plt.xlabel('V piezo')
#plt.ylabel('Intensity')
#
#intensity.plot()
#intensity_hat[max_index1_hat:max_index2_hat].plot()
#plt.plot([min_index+max_index1_hat, min_index+max_index2_hat],
#        [intensity_hat[min_index+max_index1_hat],
#            intensity_hat[min_index+max_index2_hat]],
#        marker='X', markersize=10, linestyle=' ')
#
#
## # # # # # # #
## Gennemsnit over alle V-bølgelængder
## (lamda/2) / lambda_signal ??
## # # # # # # #
#
## Corresponding piezo voltages
#V1 = v_piezo[max_index1_hat + min_index]
#V2 = v_piezo[max_index2_hat + min_index]
#
## Difference of piezo voltages (wavelength so to say)
#lambda_signal = V2 - V1
## Laser wavelength (We used a red HeNe-Laser)
#lambda_laser = 633 * 10**-9
#
#C = lambda_laser / (2 * lambda_signal)
#print(C)
#plt.show()
##
##
##
##
##
##
###""" Experimental Setup 3: Mach Zehnder Interferometer """
#### Importing data:
####data_path = os.path.join(os.getcwd(), 'DataII')
####data_files = os.listdir(data_path)
####
####data_dir = os.path.join(data_path, data_files[1])
####df = pd.read_csv(data_dir, skiprows=[1])
#####print(df.head())
#####print(df.Tid.head())
####df.set_index('Tid', inplace=True)
####df.plot()
####plt.show()
###
###
###plt.show()
