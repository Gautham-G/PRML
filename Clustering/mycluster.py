import numpy as np

def cluster(bow, K, num_iters = 100, epsilon = 1e-12):
# 	"""

# 	:param bow:
# 		bag-of-word matrix of (num_doc, V), where V is the vocabulary size
# 	:param K:
# 		number of topics
# 	:return:
# 		idx of size (num_doc), idx should be 1, 2, 3 or 4
# 	"""
    n_d = bow.shape[0]
    n_w = bow.shape[1]
    
    T = bow
    
    pi_o = np.ones(K)/K
    pi_n = pi_o
    gamma = np.zeros((n_d, n_w))
    gamma_num = 1
    
#     Normalize
    mu = np.random.rand(n_d, n_w)
    for i in range(n_d):
        mu[i][:] = mu[i][:]/np.sum(mu[i][:])
    
    iter = 0
    while(True):
        if(np.linalg.norm(pi_o-pi_n)>=epsilon or iter<num_iters):

        #     Expectation
            pi_o = pi_n
            den=[]
            for i in range(n_d):
                summ=[]
                for c in range(K):
                    M = 1
                    for j in range(n_w):
                        M = M*mu[j][c]**T[i][j]
                    summ.append(M*pi_o[c])
                den.append(np.array(summ).sum())

            for i in range(n_d):
                for c in range(4):
                    M = 1
                    for j in range(n_w):
                        M = M*mu[j][c]**T[i][j]

                    gamma[i][c] = pi_o[c]*M            
                    gamma[i][c] = gamma[i][c]/den[i]





    #      Maximization
            D = np.zeros(4)
            for c in range(4):
                for j in range(n_w):
                    for i in range(n_d):
                        D[c]+= gamma[i][c]*T[i][j]

            for c in range(4):
                for j in range(n_w):
                    N=0
                    for i in range(n_d):
                        N += gamma[i][c]*T[i][j]
                    mu[j][c] = N/D[c]

            l = np.zeros(4)
            for c in range(4):    
                for i in range(n_d):
                    l[c] += gamma[i][c]
                pi_n[c] = l[c]/n_d

            iter += 1
#             print(iter)
        else:
            break
    idx=np.zeros(n_d)
    for i in range(n_d):
        idx[i]= np.argmax(gamma[i])+1
       
    
#     raise NotImplementedError

    return idx
