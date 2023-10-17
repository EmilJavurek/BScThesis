"""
Contains core simulation files to execute the SIR function on a whole dataset across a whole parameter space.

simulation_library function executes simul function across all given (p)'s and for all given (batch)es
simul function executes SIR function for all networks (G) in given batch and for all parameter combinations (beta,gamma,rho,chi,strategy,mutation)

uses half the CPU cores.

Final Dictionary Output Structure of simulation_library:
KEY is tuple of form:
(p,beta,gamma,rho,chi,strategy name(string),mutation)           #i.e. all defining parameters of epidemic simulation
VALUE is list of SIR function outputs, that is, list of tuples of form
(t,S,I,R)                                                       #i.e. list of all outcomes of simulations for given key

"""

import multiprocessing as mp
import numpy as np
from collections import defaultdict
from collections import Counter
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import random
import gc
import warnings
import itertools
warnings.filterwarnings("ignore", category=DeprecationWarning)
from epidemic_function import *
"""
NOTES:
SIR function arguments: (G, beta, gamma, rho = 0, chi = 0.004, tmax = 10**3, strategy = arbitrary, mutation = False)
strategies: [arbitrary, highest_degree, highest_CC, lowest_CC]

makes use of Network Library:
library p values: 10**np.arange(-2,0.1,0.1)
"""



def simul(p, batch, dd, queue):
    """ Inner Simulation Function
    takes given p and batch values, loads the correct batch from the Network Library
    and runs simulations on all netowrks in batch on all parameter combinations

    Parameters:
    p (int)             : index of p_array - rewiring probability of the networks - between [0,1]
    batch (int)           : batch number - between [0,31]
    dd (dictionary) with containing:
        beta (float or iterable of floats)          : infection rate - between [0,1]
        gamma (float or iterable of floats)         : recovery rate - between [0,1]
        rho (float or iterable of floats)           : vaccination rate - between [0,1]
        chi (float or iterable of floats)           : mutation rate - between [0,1]
        strategy (function or iterable of functions): vaccination rule
        mutation (bool)                             : whether the model includes gene mutation
    queue : multiprocessing manager that handles function output

    Returns:
    None directly but puts into queue the output object - structured the same way as
    in main function (main simulation function compiles these output from all child processes into one)
    """
    filepath = f'Network_Library/p_{p}/p_{p}st_batch_{batch}.pickle'
    N = 10**4
    p_array = 10**np.arange(-2, 0.1, 0.1) #log scale constant stepsize

    output = defaultdict(list)

    with open(filepath, 'rb') as f:
        BATCH = pickle.load(f)

        for G, beta, gamma, rho, chi, strategy, mutation in itertools.product(BATCH, dd["beta"], dd["gamma"], dd["rho"], dd["chi"], dd["strategy"], dd["mutation"]):
            epidemic_output = SIR(G, beta, gamma, rho, chi, strategy, mutation)
            output[p_array[p],beta,gamma,rho,chi,strategy.__name__,mutation].append(epidemic_output)



    print(f"Successful batch {batch} for p {p} complete")
    queue.put(output)

    #memory management
    objects_to_delete = [BATCH, output]
    del objects_to_delete[:]
    gc.collect()

def transform_p_to_index(p_list):
    """
    From input of p_values finds their corresponding indices in the p_array
    that annotates files in the Network_Library
    """
    p_array = 10**np.arange(-2,0.1,0.1)
    indices = []
    for e in p_list:
        indices.append(np.where(p_array == e)[0][0])
    return list(indices)

def simulation_library(**simulation_inputs):
    """Main Simulation Function using the Network Library

    Parameters:
    p (float or iterable of floats)             : rewiring probability of the networks - between [0,1] and in p_array
    batches (int or iterable of ints)           : number of batches (of 256) of networks to be simulated - between [1,32]
    beta (float or iterable of floats)          : infection rate - between [0,1]
    gamma (float or iterable of floats)         : recovery rate - between [0,1]
    rho (float or iterable of floats)           : vaccination rate - between [0,1]
    chi (float or iterable of floats)           : mutation rate - between [0,1]
    strategy (function or iterable of functions): vaccination rule
    mutation (bool)                             : whether the model includes gene mutation


    Returns:
    final_output (dictionary):
    key is tuple of form (p,beta,gamma,rho,chi,strategy name(string),mutation) #i.e. all defining parameters of epidemic simulation
    value is list of SIR function outputs, that is, list of tuples of form (t,S,I,R) #i.e. list of all outcomes of simulations for given key



    Example use:
    # p = 10**np.arange(-2,-1.8,0.1)
    # batches = 1
    # beta = 1/4
    # gamma = 1/3
    # rho = 0.1
    # chi = 0.004
    # strategy = highest_degree
    # mutation = True
    # testing = simulation_library(p = p, batches = batches, beta = beta, gamma = gamma, rho = rho, chi = chi, strategy = strategy, mutation = mutation)
    #
    # print("##########")
    # print(testing[p[0],beta,gamma,rho,chi,strategy.__name__,mutation][:3])
    # print(testing[10**-2,1/4,1/3,0.1,0.004,"highest_degree",True][:3]) #same thing as one line above
    """
    #make every variable iterable
    for key, value in simulation_inputs.items():
        if not isinstance(value, (list, np.ndarray, range)):
            simulation_inputs[key] = [value]


    simul_inputs =  {key: simulation_inputs[key] for key in simulation_inputs.keys() - {'p', 'batches'}}

    p_indices = transform_p_to_index(simulation_inputs["p"])

    num_cpus = mp.cpu_count() // 2  # use half the CPU cores
    with mp.Pool(num_cpus) as pool:
        queue = mp.Manager().Queue()
        inputs = [(i, j, simul_inputs, queue) for i in p_indices for j in simulation_inputs["batches"]]
        pool.starmap(simul, inputs)

    #collect results from the queue
    final_output = defaultdict(list)
    while not queue.empty():
        output = queue.get()
        for key, value in output.items():
            final_output[key].extend(value)

    return final_output
