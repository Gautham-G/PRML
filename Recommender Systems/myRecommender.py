import numpy as np

# Name : Gautham Gururajan

def my_recommender(rate_mat, lr, with_reg):
    
    
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]
    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr
    
    # Initialize convergence criteria

    it = 0
    thresh = 0.001
    diff = 1e6
    max_iter = 500
    
    # Vectorized form of the GD
    if(with_reg):
        reg_coef = 0.02    
        learning_rate = 0.0002

    else:
        reg_coef = 1
        learning_rate = 0.0002
 
    while (it<max_iter and diff>thresh):
        A = rate_mat>0
        U = U+2*learning_rate*((rate_mat-np.multiply(U@V.T, A))@V) -2*learning_rate*reg_coef*U
        V = V+2*learning_rate*((rate_mat-np.multiply(U@V.T, A)).T@U) -2*learning_rate*reg_coef*V
            
        # For convergence criteria, keeping just a criteria based on iterations actually reduces the run-time since it is costly to compute this
        diff = np.sum(np.power((rate_mat-np.multiply(U@V.T, A)), 2))+reg_coef*np.sum(np.power(U, 2))+reg_coef*np.sum(np.power(V, 2))
        it+=1
        # print(it)

    return U, V