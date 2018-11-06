#Disclaimer : The algorithm was implemented here to study its behaviour, and was used on a graph automatically generated 
# by using the networkx package from python. 

#QUESTIONS ANSWERED : 

# =============================================================================
# Question 1.1
# 
# PageRank is a method for ranking nodes. Explain why it is so popular and why it is useful.
# Question 1.2
# 
# Implement iterative and exact algebraic versions of the PageRank algorithm. Use an undirected Barabasi-Albert random network on 1000 nodes and compare the answers using Mean Squared Error (L2 norm of the difference between true and estimated vectors). Use α=0.85
# 
# .
# Question 1.3
# 
# What are the advantages of the iterative versus the exact methods for calculating PageRank?
# Question 1.4
# 
# Using the iterative version, calculate the PageRank for a range of α
# values between 0 and 1. How does the choice of alpha influence the results (the overall ranking and the number of iterations)? Why do you think α=0.85
# 
# is a popular choice?
# Question 1.5
# 
# List the problems that can arise when calculating the PageRank on directed graphs and explain how you might deal with these in practice. Generate a directed scale-free graph and show how the pageRank changes between the directed and undirected versions of the graph. Explain your observations.
# =============================================================================
    
# 1.1 #######################################
# =============================================================================
# 1.1	Pagerank is a very useful algorithm that is used to assign a rank to each nodes 
# by the use of a random surfer on the graph. We just have to look at how well Google 
# performed back when they first implemented it. To prevent the surfer to get stuck and 
# make it switch from node to node more smoothly we had a low probability (alpha/n) to 
# jump from current node to any other node with an uniform probability. 
# The algorithm is very useful for search engines where they assign a rank to each page to know 
# which pages should be shown first in the search. It also provides a fast algorithm for approximating the PageRank vector. 
# =============================================================================


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

# 1.2 ############################### Comments
# The matrix we have used in eigenvector computation is of "high" dimension (not quite like the ones used by Google but still) and that 
# makes the spectral decomposition hard to compute because we have a lot of 0's in the matrix. Thus the values are not exactly the same (at least that is my interpretation of why they differ).
# But overall, an mse of 0.00000008 seems small enough to say that both our implementations are very similar, even if the iteration method seems better suited for high dimension work.
# Also, even tough our google matrix is stochastic (as we checked for it) is 1.41, not 1. The property says that the largest value should be 1, thus showing that the spectral decomposition wasn't 
# performed 100% accurately, however i did not find a way to compensate for this other than normalizing the vector by their L1 norm.

# 1.3 ###############################
# =============================================================================
# The power method (iterative method) had some advantages, like the parameter alpha for convergence rate, 
# the convergence is independant of matrix dimension, it only stores in a single vector. It is also accurate (no substractions) 
# and very simple. 
# The matrix that will be dealt with will be immense because we are analyzing millions of pages, 
# thus we need an efficient way to calculate the eigenvector of a square matrix with very high dimension. 
# It is very effective because we do not have to compute a matrix decomposition which is nearly impossible for 
# high dimensionnal matrices with very few values. The downside is that we are only able to compute the largest eigenvector with 
# the iterative method. We can also say that having control on alpha is very good because it allows us to have control on the 
# speed of convergence. 
# =============================================================================

# 1.4 ###############################    


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

alphaList = [0.1,0.3,0.5,0.85] #Number of iterations needed : [5,7,8,10]
mseAlpha = []

    
for alpha in alphaList:
    mseAlpha.append(pageRankItEpsilon(G, alpha))
print("iterations : \n")

for i in range(0,4):
    print(alphaList[i], "=", mseAlpha[i][1])
    

#Checking the results, it seems that the order of the pages wont change much but some things are noticeable here. 
#First, the values tend to be bigger as alpha grows.
#Second, the gap between low values and higher values is bigger, this could help with the ranking.
#Third, the number of iterations go up as alpha grows so it would mean more time to process but it seemed reasonable as the iterations went from 5 to 10 (0.1 vs 0.85). 
#0.85 is probably a popular choice because of these reasons. 
#Alpha/n gives a smoother walk as we add the teleportation
    

# 1.5 #################################
# Spectral theory is different, asymmetric matrix may not be diagonalizable.
# Eigenvectors may not be orthogonal
# probability of vertices can vary
# Very slow to converge to distribution 

