"""
Contains Core functions for the epidemic spread simulation, optimized for speed.

SIR function
-simulates the SIR epidemic spread model on a network. Allows for mutation of infectivity and for vaccination according to a strategy.

Supporting functions:

-Vaccination strategy functions:
--arbitrary (uniform random)
--highest_degree
--highest_CC (clustering coefficient)
--lowest_CC

-test_infection (executes probabilistic infection)
-mutation_test (executes probabilistic mutation)

"""

import networkx as nx
import numpy as np
import random
from collections import defaultdict
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#vaccination strategies:
def arbitrary(G, vaccinations):
    """Vaccination Strategy, choses nodes to be vaccinated and selects first infected node
    node selection for vaccination is arbitrary (random)

    Parameters:
    G (networkx Graph): graph from which nodes for vaccination are selected
    vaccinations (int): number of nodes to be vaccinated

    Returns:
    vaccinated (list): list of vaccinated nodes
    patient0 (node): the initial infected node
    """
    if vaccinations == G.order():
        vaccinations += -1

    draw = random.sample(G.nodes(), vaccinations + 1) #list(G.nodes) to avoid warning !BUT! Slows code by 1%
    vaccinated = draw[:-1]
    patient0 = draw[-1]
    return vaccinated, patient0

def highest_degree(G, vaccinations):
    """Vaccination Strategy, choses nodes to be vaccinated and selects first infected node
    node selection for vaccination is by degree (highest first)

    Parameters:
    G (networkx Graph): graph from which nodes for vaccination are selected
    vaccinations (int): number of nodes to be vaccinated

    Returns:
    vaccinated (list): list of vaccinated nodes
    patient0 (node): the initial infected node
    """
    if vaccinations == G.order():
        vaccinations += -1
    if vaccinations == 0:
        return [], random.sample(G.nodes(), 1)[0]

    sort_by_degree = sorted(G.degree, key=lambda x: x[1], reverse=True)
    vaccinated = [u[0] for u in sort_by_degree[:vaccinations]]
    patient0 = [random.sample(sort_by_degree[vaccinations:], 1)[0][0]][0] #list(G.nodes) to avoid warning !BUT! Slows code by 1%
    return vaccinated, patient0

def highest_CC(G, vaccinations):
    """Vaccination Strategy, choses nodes to be vaccinated and selects first infected node
    node selection for vaccination is by value of CC (highest first)

    Parameters:
    G (networkx Graph): graph from which nodes for vaccination are selected
    vaccinations (int): number of nodes to be vaccinated

    Returns:
    vaccinated (list): list of vaccinated nodes
    patient0 (node): the initial infected node
    """
    if vaccinations == G.order():
        vaccinations += -1
    if vaccinations == 0:
        return [], random.sample(G.nodes(), 1)[0]

    CC = nx.clustering(G)
    c = Counter(CC)
    sort_by_CC = c.most_common()
    vaccinated = [u[0] for u in sort_by_CC[:vaccinations]]
    patient0 = [random.sample(sort_by_CC[vaccinations:], 1)[0][0]][0]
    return vaccinated, patient0

def lowest_CC(G, vaccinations):
    """Vaccination Strategy, choses nodes to be vaccinated and selects first infected node
    node selection for vaccination is by CC (lowest first)

    Parameters:
    G (networkx Graph): graph from which nodes for vaccination are selected
    vaccinations (int): number of nodes to be vaccinated

    Returns:
    vaccinated (list): list of vaccinated nodes
    patient0 (node): the initial infected node
    """
    if vaccinations == G.order():
        vaccinations += -1
    if vaccinations == 0:
        return [], random.sample(G.nodes(), 1)[0]

    CC = nx.clustering(G)
    c = Counter(CC)
    sort_by_CC = c.most_common()
    vaccinated = [u[0] for u in sort_by_CC[-vaccinations:]]
    patient0 = [random.sample(sort_by_CC[:-vaccinations], 1)[0][0]][0]
    return vaccinated, patient0

def test_infection(beta, mutation, alpha):
    """Samples probability to determine whether infection occurs.
    Handles both when mutation is True/False

    Parameters:
    beta (float): infection rate when mutation False
    mutation (bool): whether the underlying model has a mutation virus
    alpha (int): the gene of the virus that is trying to infect

    Returns:
    Bool : transmission of infection occurence
    """
    if mutation:
        p = [0.45, 0.35, 0.95] #for alpha [-1,0,1], respectively
        return random.random() < p[alpha+1]
    else:
        return random.random() < beta

def mutation_test(alpha, chi):
    """Performs mutation of virus (alpha) according to said probabilities
    Essentialy one step on a discrete 3-state Markov Chain

    Parameters:
    alpha (int): current gene of virus
    chi (float): mutation probability

    Returns:
    int: new alpha value
    """
    P_neg1 = [1-chi, chi, 0]
    P_zero = [chi/2, 1-chi, chi/2]
    P_pos1 = [0, chi, 1-chi]
    transition_matrix = np.array([P_neg1, P_zero, P_pos1])

    states = [-1, 0 ,1]
    current_i = alpha + 1

    return np.random.choice(states, p = transition_matrix[current_i])

def SIR(G, beta, gamma, rho = 0, chi = 0.004,strategy = arbitrary, mutation = False, tmax = 10**3):
    """Performs the simulation of an epidemic on a network
    Virus spreads in discrete timesteps. Each step, first the virus spreads, then infected (possibly) recover and then,
    if activated, gene (possibly) mutates.
    During transmission the newly infected initially inherits the gene of the virus that infected them

    Parameters:
    G (networkx Graph): underlying network on which epidemic occurs
    beta (float): infection rate - between [0,1]
    gamma (float): recovery rate - between [0,1]
    rho (float): vaccination rate - between [0,1]
    chi (float): mutation rate - between [0,1]
    strategy (function): vaccination rule
    mutation (bool): whether the model includes gene mutation
    tmax(int): limit of timesteps

    Returns:
    t (list): timesteps
    S (list): susceptible per timestep
    I (list): infected per timestep
    R (list): recovered per timestep
    """
    N = G.order()

    susceptible = defaultdict(lambda: True)
    vaccinations = int(round(N*rho))

    #select vaccinated
    vaccinated, patient0 = strategy(G, vaccinations)
    #vaccinate:
    for u in vaccinated:
        susceptible[u] = False
    #inital infection
    susceptible[patient0] = False

    alpha = defaultdict(lambda: -1) #relevant if mutation True


    #outputs
    t = [0]
    S = [N-1-vaccinations]
    I = [1]
    R = [vaccinations]


    infected = set([patient0])
    while infected and t[-1]<tmax:
        R.append(R[-1]) #once recovered always recovered

        new_infected = set()
        for u in infected:
            #infection
            for v in G.neighbors(u):
                if susceptible[v] and test_infection(beta, mutation, alpha[u]):
                    new_infected.add(v)
                    susceptible[v] = False
                    alpha[v] = alpha[u] #inherit mutation

            #recovery
            if random.random() < gamma:
                R[-1] += 1
            else:
                new_infected.add(u)

        #mutation
        if mutation:
            for u in new_infected:
                alpha[u] = mutation_test(alpha[u],chi)



        infected = new_infected

        I.append(len(infected))
        S.append(N-I[-1]-R[-1])
        t.append(t[-1]+1)

    return t,S,I,R #I want to return some info about mutations too
