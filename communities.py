# 2.1 #################################################
#We can see on the graph that our communities are well separated, as expected with Pout = 0.001

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community as cm 
import operator

#generating a community graph
G = nx.random_partition_graph([10,15,25],1,0.001)
nx.draw(G)

S = nx.to_numpy_matrix(G) #Numpy adj matrix

plt.imshow(S) #show adjacency matrix

#2.2 ##################################################

def create_graph_k_cliques(k):
    groups = []
    for i in range(0, k):
        groups.append(10)
    G = nx.random_partition_graph(groups, 1, 0.001)
    return G

def plotG(G):
    S = nx.to_numpy_array(G)
    plt.imshow(S)
    
Gtwo = create_graph_k_cliques(10)
plotG(Gtwo)

#2.3 #################################################

graphs = []
louvain = []
for i in range(2, 151):
    graphs.append(create_graph_k_cliques(i))
    louvain.append(cm.best_partition(graphs[i-2]))

numbComm = []
for i in range(0, len(louvain)):
    numbComm.append((max(louvain[i].items(), key=operator.itemgetter(1))[1]) + 1) #maximum of dictionary + 1 for community "0"

print(numbComm)

# =============================================================================
# We see that k = number of communities. This is probably because P_out is 0.001 so when the cliques are generated they have very little chance
# of having a link between communities. 
# =============================================================================
