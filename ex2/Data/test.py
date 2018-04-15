import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# Directing datafiles 
data_path = os.getcwd() + '/Data/'
data_file = os.listdir(data_path)
data = data_path + data_file[2]


mydata = pd.read_csv(data, header=3, names=['Time [ms]', 'ChanA [mV]', 'ChanB [V]'])

#print(df[['ChanB [V]']].head())
#print(df.head())
#print(np.array(df.head()))
#
# df.to_csv('name.csv')

fig = plt.figure()
mydata.plot(x='Time [ms]', y='ChanA [mV]', style='-')
mydata.plot(x='Time [ms]', y='ChanB [V]', style='-')

plt.show()
