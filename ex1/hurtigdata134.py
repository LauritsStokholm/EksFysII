import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import
data = os.listdir()
csv_file = [x for x in data if '.csv' in x]
print(csv_file)
data_path = os.path.join(os.getcwd(), csv_file[1])
print(data_path)



df = pd.read_csv(data_path, skiprows=[0], names=['Degrees', 'Signal'], delimiter='.')
print(df.head())