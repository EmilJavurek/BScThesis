"""
Collects output from Data_creation.py into one final data file containing one dictionary.


"""
import numpy as np
import pickle
from random import random
from collections import defaultdict

total_data = defaultdict(list)

p_array = 10**np.arange(-2,0.1,0.1)
batch_array = range(32)
batches = batch_array[4:8] #to save runtime
for p in range(len(p_array)):
    filepath = f"Simulation_Results/p_{p}_Batches_{batches}.pickle"
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        for key, value in data.items():
            total_data[key].extend(value)


endpath = f"Data_final/Data_batches_{batches}.pickle"
pickle.dump(total_data, open(endpath, 'wb'))
