# 1.2 #######################################
#First is iterative method, then exact method

import numpy as np 
import networkx as nx
from scipy import linalg

nNodes = 1000
G = nx.barabasi_albert_graph(nNodes, 25, seed = 1234)
#G is the generated graph
#S is the matrix representing links
def pageRankIt(G, alpha = 0.85, K = 2000):
    nNodes = nx.number_of_nodes(G)

    #Creating the network
    S = nx.to_numpy_matrix(G) #Numpy adj matrix
    
    for i in range(0, nNodes): 
        S[i, :] = S[i, :] / np.sum(S[i, :])
    summ = np.sum(S, axis = 1) #check == 1, probability that surfer goes from i to j
    
    v = np.random.rand(nNodes, 1) #initial guess
    v = v / np.linalg.norm(v, 1) #L1
    
    #google matrix 
    GM = (alpha * S) + (((1 - alpha) / nNodes) * np.matmul(np.ones(nNodes), v))
    summG = np.sum(GM, axis = 1) #check == 1, stochastic matrix
    
    for i in range(0, nNodes):
        GM[i, :] = GM[i, :] / np.sum(GM[i, :])
    
    v = v.transpose()
    
    for i in range(0, K):
        v = np.matmul(v, GM)
        
    v = v.transpose() #put it back in column
    
    #Normalize vector by its norm
    v = v/np.linalg.norm(v, 1)

    return v
    

    
def pageRankEig(G, alpha = 0.85):
#real pagerank vector ? = google matrix eigenvector 
    nNodes = nx.number_of_nodes(G)
    Mat = nx.to_numpy_matrix(G) #Numpy adj matrix
    
    #S matrix
    S = Mat
    
    for i in range(0, nNodes):
        S[i, :] = S[i, :] / np.sum(S[i, :])
    summ = np.sum(S, axis = 1) #check == 1, probability that surfer goes from i to j
    
    v = np.random.rand(nNodes, 1) #initial guess
    v = v / np.linalg.norm(v, 1) #L1
    
    #google matrix 
    GM = (alpha * S) + (((1 - alpha) / nNodes) * np.matmul(np.ones(nNodes), v))
    summG = np.sum(GM, axis = 1) #check == 1, stochastic matrix
    
    for i in range(0, nNodes):
        GM[i, :] = GM[i, :] / np.sum(GM[i, :])
    eigValue, eigVector = linalg.eigh(GM, check_finite = True)
    eigV = eigVector[:,nNodes - 1]
    eigV = eigV * -1 #positive pr
    
    #Normalize vector by its norm so it equals iteration method
    eigV = eigV / np.linalg.norm(eigV, 1)
    
    return eigV
    
v = pageRankIt(G)
eigV = pageRankEig(G)

#Length must be = 
def MSE(eigV, v, N):
    mseArr = np.zeros(N)
    for i in range(0, nNodes):
        mseArr[i] = (eigV[i] - v[i])**2
    mse = np.sum(mseArr)/nNodes #8.729242231093588e-08
    return mse

mse = MSE(eigV, v, nNodes)


#1.4 ###############################    


#Epsilon is the threshold
def pageRankItEpsilon(G, alpha = 0.85, eps = 0.00000001):
    nNodes = nx.number_of_nodes(G)
    iterationsCounter = 0
    
    #Creating the network
    S = nx.to_numpy_matrix(G) #Numpy adj matrix
    
    for i in range(0, nNodes): 
        S[i, :] = S[i, :] / np.sum(S[i, :])
    summ = np.sum(S, axis = 1) #check == 1, probability that surfer goes from i to j
    
    v = np.random.rand(nNodes, 1) #initial guess
    v = v / np.linalg.norm(v, 1) #L1
    last_v = np.ones((nNodes, 1), dtype=np.float32) * 1000 #initial pass for epsilon 
    #google matrix 
    GM = (alpha * S) + (((1 - alpha) / nNodes) * np.matmul(np.ones(nNodes), v))
    summG = np.sum(GM, axis = 1) #check == 1, stochastic matrix        
    
    for i in range(0, nNodes):
        GM[i, :] = GM[i, :] / np.sum(GM[i, :])
        
    v = v.transpose()
    
    while np.linalg.norm(v-last_v, 2) > eps:
        iterationsCounter += 1
        last_v = v
        v = np.matmul(v, GM)
        
    v = v.transpose() #put it back in column
    
    #Normalize vector by its norm
    v = v/np.linalg.norm(v, 1)
    
    #Prepare list for return 
    ret = [v, iterationsCounter] 
    
    return ret

alphaList = [0.1,0.3,0.5,0.85]
mseAlpha = []

for alpha in alphaList:
    mseAlpha.append(pageRankItEpsilon(G, alpha))
    
#Checking the results, it seems that the order of the pages wont change much but some things are noticeable here. 
#First, the values tend to be bigger as alpha grows. 
#Second, the gap between low values and higher values is bigger, this could help with the ranking.
#Third, the number of iterations go up as alpha grows so it would mean more time to process. 
    
####################################
#The matrix we have used in eigenvector computation is of "high" dimension (not quite like the ones used by Google but still) and that 
#makes the spectral decomposition hard to compute because we have a lot of 0's in the matrix. Thus the values are not exactly the same (at least that is my interpretation of why they differ).
#But overall, an mse of 0.00000008 seems small enough to say that both our implementations are very similar, even if the iteration method seems better suited for high dimension work.
#Also, even tough our google matrix is stochastic (as we checked for it) is 1.41, not 1. The property says that the largest value should be 1, thus showing that the spectral decomposition wasn't 
#performed 100% accurately, however i did not find a way to compensate for this other than normalizing the vector by their L1 norm.




