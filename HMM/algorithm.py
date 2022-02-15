import matplotlib.pyplot as plt
import numpy as np


def algo(q, Y):
    # init
    p = 0.0

    
#     For the forward pass see as below
    num_weeks = len(Y)
    emission_mat = np.array([[q, 1-q], [1-q, q]])
    transition = np.array([[0.8, 0.2], [0.2, 0.8]])
    pi = [0.2, 0.8]
    
    alpha = np.zeros((num_weeks, 2))
    alpha[0][0]=pi[0]*emission_mat[1][0]
    alpha[0][1]=pi[1]*emission_mat[1][1]
#     print(alpha)
    
    for week in range(2, num_weeks+1):
        if(Y[week-1]==1):
            alpha[week-1,0]=emission_mat[0,0]*(alpha[week-2,0]*transition[0,0]+alpha[week-2,1]*transition[1,0])
            alpha[week-1,1]=emission_mat[0,1]*(alpha[week-2,0]*transition[0,1]+alpha[week-2,1]*transition[1,1])
        else:
            alpha[week-1,0]=emission_mat[1,0]*(alpha[week-2,0]*transition[0,0]+alpha[week-2,1]*transition[1,0])
            alpha[week-1,1]=emission_mat[1,1]*(alpha[week-2,0]*transition[0,1]+alpha[week-2,1]*transition[1,1])
#     print(alpha)
            
            
    p1=alpha[num_weeks-1,0]+alpha[num_weeks-1,1];
    
    
 #     For the backward pass see as below
   
    beta = np.zeros((num_weeks, 2))
    beta[num_weeks-1, 0] = 1.0
    beta[num_weeks-1, 1] = 1.0
    
    
    for week in range(num_weeks-1, 0, -1):
        if(Y[week]==1):
            beta[week-1,0]=transition[0, 0]*emission_mat[0, 0]*beta[week, 0] + transition[0, 1]*emission_mat[0, 1]*beta[week, 1] 
            beta[week-1,1]=transition[1, 0]*emission_mat[0, 0]*beta[week, 0] + transition[1, 1]*emission_mat[0, 1]*beta[week, 1]
        else:
            beta[week-1,0]=transition[0, 0]*emission_mat[1, 0]*beta[week, 0] + transition[0, 1]*emission_mat[1, 1]*beta[week, 1]
            beta[week-1,1]=transition[1, 0]*emission_mat[1, 0]*beta[week, 0] + transition[1, 1]*emission_mat[1, 1]*beta[week, 1]
    
#     print(alpha)
#     print(beta)
    p2 = alpha*beta
    
#     print(p1)
#     print(p2.shape)
    
    p=p2/p1
#     p = p[:, 0]
    fig, ax = plt.subplots()
    ax.plot(p[:, 0])
#     fig = plt.plot(p[:, 0])
    # TODO implement your algorithm and return the (i) prob p and (ii) a matplotlib Figure object for the plot

    return p, fig