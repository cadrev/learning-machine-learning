import random
import numpy as np
from sklearn.datasets.samples_generator import make_regression 

def grad_desc(x,y, alpha, err, max_iter):
    m           = y.shape[0]
    iteration   = 0
    convergence = False

	# Sets the initial theta for model
    theta0     = np.random.random()
    theta1     = np.random.random()

    # J(theta) function
    total_err  = sum([(theta0 + theta1*x[i] - y[i])**2 for i in range(m)])

    while not convergence:

        # Compute the gradient for each theta
        gradient0 = (1.0/m) * sum([(theta0 + theta1*x[i] - y[i])      for i in range(m)]) 
        gradient1 = (1.0/m) * sum([(theta0 + theta1*x[i] - y[i])*x[i] for i in range(m)])

        # t0,t1 placeholder for the computed theta
        t0     = theta0 - alpha * gradient0
        t1     = theta1 - alpha * gradient1
        theta0 = t0
        theta1 = t1
        
        # compute the mean squared error
        mse    = sum([(theta0 + theta1*x[i] - y[i])**2 for i in range(m)])
        
        if abs(total_err - mse) <= err:
            convergence = True
        
        if iteration == max_iter:
        	convergence = True
        
        total_err = mse
        iteration = iteration + 1
  
    #return the slope(theta0), intercept(theta1) and number of iterations
    return theta0, theta1, iteration

# Check the performance the gradient descent function
def main():
  
    # load the dataset to the two variables
    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35) 
    
    # criteria for the gradient descent
    learning_rate        = 0.1
    convergence_criteria = 0.01 
    
    # get the slope
    slope, intercept, iterations = grad_desc(x, y, learning_rate, convergence_criteria, 1000)

    print 'slope: ' + str(slope)
    print 'intercept: ' + str(intercept)
    print 'number of iterations: ' + str(iterations)



if __name__ == '__main__':
 	main() 
