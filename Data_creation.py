"""
Runs the simulation on the given part of Network library and so produces a data set.
Data is saved in a Simulation_Results folder, in files per each value of p.
k
"""
from epidemic_function import * #my file
from simulation import * #my file
import numpy as np
import pickle
from itertools import product
from random import random
from time import time
import gc
import os

if __name__ == '__main__':

    #ranges
    p_array = 10**np.arange(-2,0.1,0.1)
    batch_array = range(32)
    beta = 0.45
    gamma = 1
    rho_array = np.arange(0,1,0.1)
    chi = 0.004 #extreme is 0.01
    strat_array = [arbitrary, highest_degree, highest_CC, lowest_CC]
    mutation_array = [True,False]

    batches = batch_array[4:8] #to save runtime

    #create simulation results folder to store output
    current_dir = os.getcwd()
    final_dir = os.path.join(current_dir, r'Simulation_Results')
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)


    def create_data_for_p(i):
        s = time()
        data =  simulation_library(p = p_array[i],
                                   batches = batches,
                                   beta = beta,
                                   gamma = gamma,
                                   rho = rho_array,
                                   chi = chi,
                                   strategy = strat_array,
                                   mutation = mutation_array)
        filepath = f"Simulation_Results/p_{i}_Batches_{batches}.pickle"
        pickle.dump(data, open(filepath, 'wb'))
        e = time()
        del data
        gc.collect()
        print(f"{i+1}th p done, runtime: {e-s}")

    for i in range(0,len(p_array)):
        create_data_for_p(i)
