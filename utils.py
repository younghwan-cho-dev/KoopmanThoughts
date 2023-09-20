import numpy as np
from itertools import combinations_with_replacement

def centralDifference(arr, dt):
    arr_shifted_forward = shift_for_np(arr,1)
    arr_shifted_backward = shift_for_np(arr,-1)
    df = arr_shifted_backward - arr_shifted_forward
    return (df/(2*dt))[1:-1] # The number of data points shrinks to n-2.

def shift_for_np(arr, num):
    arr=np.roll(arr,num)
    if num<0:
         np.put(arr,range(len(arr)+num,len(arr)),np.nan)
    elif num > 0:
         np.put(arr,range(num),np.nan)
    return arr

def integrator_RK4(func, dt, t0, x0):

     f1 = func(t0,x0)
     f2 = func(t0+(dt/2), x0+(dt/2)*f1)
     f3 = func(t0+(dt/2), x0+(dt/2)*f2)
     f4 = func(t0+dt, x0+dt*f3)
     
     x1 = x0 + (dt/6)*(f1+(2*f2)+(2*f3)+f4)
     return x1

def func_generator(coef,poly,subset):
     
     def func(t,x0):
          X = poly.fit_transform(x0.reshape(1,-1))[0][list(subset)]
          dX = coef @ X.T
          return dX
     return func

# Work in Progress
# Need to check the logic works well
def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new

# Need to check the logic works well
def standardize(X):
    """ Standardize the dataset X """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std
