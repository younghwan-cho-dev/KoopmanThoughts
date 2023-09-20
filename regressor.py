
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
    if rho < -lamda:
        return (rho + lamda)
    elif rho >  lamda:
        return (rho - lamda)
    else: 
        return 0
    
def coordinate_descent_lasso(theta, X, y, reg_term=1, num_iters=1000, include_bias = False):
    '''Coordinate gradient descent for lasso regression - for unnormalized data. '''
    
    #Initialisation of useful values 
    N, M = X.shape

    Z = np.linalg.norm(X, axis = 0) 

    print(np.linalg.norm(X, axis = 0))

    #Looping until max number of iterations
    for i in range(num_iters): 
        #Looping through each coordinate
        for j in range(M):
            
            #Vectorized implementation
            y_pred = X @ theta
            rho = X[:,j].T @ (y - y_pred  + theta[j]*X[:,j])
            
            #Checking intercept parameter
            if include_bias == True:  
                if j == 0: 
                    theta[j,:] =  rho 
                else:
                    theta[j,:] =  soft_threshold(rho, reg_term)  

            if include_bias == False:

                theta[j,:] =  soft_threshold(rho, reg_term)   
            
    return theta.flatten()

    