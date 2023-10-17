"""
Will fill the Network Library with the networks.

In each folder, named p_0 to p_20, a pickle file is created for each batch.
Each batch is a list (of length = batch_size) of connected watts strogatz graphs with p = p_i (p_i = name of folder)
p_i's correspond to i-th indexes of p's in array p_array = 10**np.arange(-2,0.1,0.1)
Those are 21 equidistant points on the logarithmic scale between 0.01 and 1 (inclusive).

Note, in thesis only batches indexed 4,5,6,7 are used due to computational constraints.

"""
import networkx as nx
from tqdm import tqdm
import random
import pickle
import numpy as np

N = 10**4
K = 4
batches = 32 #decrease to reduce size
batch_size = 256 #decrease to reduce size

random.seed(13331124)

p_array = 10**np.arange(-2,0.1,0.1)

for index in range(21):
    p = p_array[index]

    for batch in range(batches):
        filepath = f"Network_Library/p_{index}/p_{index}st_batch_{batch}.pickle"
        print(filepath)
        collection = []
        for _ in tqdm(range(batch_size)):
            collection.append(nx.connected_watts_strogatz_graph(N, K, p, tries = N))
        #save
        pickle.dump(collection, open(filepath, 'wb'))

#to load the files:
#K = pickle.load(open(filepath, 'rb'))
