#
# Simple cost function implementation.
# Based on Andrew Ng's Standford 
# Machine Learning Class
#


# theta0 and theta1 are usually randomized.
# For this application, we set theta0 and
# theta1 manually in order to answer the
# question for the Week1 of the Machine
# learning class.
def cost_function(x,y,theta0,theta1): 
    m = len(x)  
    return sum([(theta0 + theta1*x[i] - y[i])**2 for i in range(m)]) / (2.0 * m)



def main():
    x  = [3,1,0,4]
    y  = [2,2,1,3]
    t0 = 0
    t1 = 1
    print cost_function(x,y,t0,t1)


if __name__ == '__main__':
	main()
