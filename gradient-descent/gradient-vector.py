'''
A Vector implementation of Gradient Descent

'''

import random
import numpy as np
from sklearn.datasets.samples_generator import make_regression 



def grad_desc_vector(X,y, alpha, max_iter):

# We take advantage of the numpy matrix operations
# to speed up the gradient descent computation
    m       = y.shape[0]
    theta   = np.ones(X.shape[1])
    X_trans = X.T 

    for loop in range(max_iter):
        loss     = (np.dot(X, theta) - y)
        J        = np.sum(loss ** 2) / (2 * m)
        gradient = np.dot(X_trans, loss) / m         
        theta    = theta - alpha * gradient  # update

    return theta



def main():
    
    # load the dataset to the two variables
    X, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35) 
    
    m = np.shape(X)[0]
  
    X = np.c_[ np.ones(m), X]

    # get the slope
    theta = grad_desc_vector(X, y, 0.01,  1000)

    print theta   
    


if __name__ == '__main__':
    main() 
