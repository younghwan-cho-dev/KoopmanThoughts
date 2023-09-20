
import numpy as np

def STLS_algo(X, y, treshold, n_iter):

    # Assuming Xi, Theta, dXdt, lambda, and n are defined appropriately
    N, M = y.shape
    # Compute Sparse regression: sequential least squares
    Xi = np.linalg.lstsq(X, y, rcond=None)[0]  # initial guess: Least-squares
    # lambda is our sparsification knob.
    for k in range(n_iter):
        smallinds = np.abs(Xi) < treshold
        Xi[smallinds] = 0
        for ind in range(M): # For each state...
            biginds = ~smallinds[:, ind]
            # find small coefficients and threshold
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = np.linalg.lstsq(X[:, biginds], y[:, ind], rcond=None)[0]

    return Xi

def soft_threshold(rho,lamda):
    '''Soft threshold function used for lasso regression'''
    if rho < -lamda/2:
        return (rho + lamda/2)
    elif rho >  lamda/2:
        return (rho - lamda/2)
    else: 
        return 0
    
def coordinate_descent_lasso(theta, X, y, reg_term=1, num_iters=100, include_bias = False):
    '''Coordinate gradient descent for lasso regression - for unnormalized data. '''
    
    #Initialisation of useful values 
    N, M = X.shape

    Z = np.sum(X**2,axis=0)

    #Looping until max number of iterations
    for i in range(num_iters): 
        #Looping through each independent variable
        for j in range(M):
            Z_j = Z[j]
            #Vectorized implementation
            y_pred = X @ theta
            rho = (X[:,j].reshape(-1,1).T @ (y - y_pred  + theta[j]*X[:,j].reshape(-1,1)))[0][0] # 2dim np.array's value

            #Checking intercept parameter
            if include_bias == True:  
                if j == 0: 
                    theta[j,:] =  rho 
                else:
                    theta[j,:] =  soft_threshold(rho, reg_term) / Z_j 

            if include_bias == False:

                theta[j,:] =  soft_threshold(rho, reg_term) / Z_j
            
    return theta.flatten()

    