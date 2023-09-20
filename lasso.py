
import numpy as np

def soft_threshold(rho,lamda):
    '''Soft threshold function used for normalized data and lasso regression'''
    if rho < - lamda:
        return (rho + lamda)
    elif rho >  lamda:
        return (rho - lamda)
    else: 
        return 0

def coordinate_descent_lasso(theta, X, y, reg_term=1, num_iters=1000, include_bias = False):
    '''Coordinate gradient descent for lasso regression - for normalized data. 
    The intercept parameter allows to specify whether or not we regularize theta_0'''
    
    #Initialisation of useful values 
    N, M = X.shape
    X = X / (np.linalg.norm(X,axis = 0)) #normalizing X in case it was not done before
    
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
                    theta[j] =  rho 
                else:
                    theta[j] =  soft_threshold(rho, reg_term)  

            if include_bias == False:
                theta[j] =  soft_threshold(rho, reg_term)   
            
    return theta.flatten()

    