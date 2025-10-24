import numpy as np 
import matplotlib.pyplot as plt

d = 50   #feature dimension

X = np.load('./data/X.npy')
y = np.load('./data/y.npy')
print ("data shape: ", X.shape, y.shape)

theta_star = np.load('./data/theta_star.npy')

###### part (1): least square estimator ########


# theta_hat = (X^T X) ^ {-1} X^T y
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
# theta_hat = None # TODO: calculate the least square solution

Error_LS = np.linalg.norm(theta_hat - theta_star, 2)
print('Estimator approximated by LS:',Error_LS)
# print("theta_hat:", theta_hat)
# print("theta_star:", theta_star)


###### part (2): L1 estimator ########
mu = 1e-5  # smoothing parameter
alpha = 0.001  # stepsize
T = 1000  # iteration number

# random initialization
theta = np.random.randn(d,1)

Error_huber = []

for _ in range(1, T):

    # calculate the l2 error of the current iteration
    Error_huber.append(np.linalg.norm(theta-theta_star, 2)) 

    # TODO: calculate gradient -- vector of 1*50
    grad = np.sign(theta - theta_star)

    #gradient descent update
    theta = theta - alpha * grad
    
#######   plot the figure   #########
plt.figure(figsize=(10,5))
plt.yscale('log',base=2) 
plt.plot(Error_huber, 'r-')
plt.title(r'$\ell_1$ estimator approximated by Huber')
plt.ylabel(r'$\theta$')               # set the label for the y axis
plt.xlabel('Iteration')              # set the label for the x axis
# plt.show()
plt.savefig('l1_estimator.png')