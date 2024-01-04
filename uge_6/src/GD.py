import numpy as np
import matplotlib.pyplot as plt

# Load the data
X = np.loadtxt("X_data.csv", delimiter=',')
y = np.loadtxt("y_data.csv", delimiter=',')

# Set λ and other parameters
lambda_val = 5
eta = 1e-4  # Learning rate
num_iterations = 5000  # Number of iterations

# Initialize beta randomly or with zeros
d = X.shape[1]  # Number of features
beta = np.zeros(d)  # Initialize beta with zeros

# Function to calculate the logistic loss with L2 regularization
def logistic_loss(beta, X, y, lambda_val):
    n = len(y)
    logits = X @ beta
    loss = np.log(1 + np.exp(logits)) - y * logits
    reg_term = 0.5 * lambda_val * np.linalg.norm(beta)**2
    return np.mean(loss) + reg_term

# Function to calculate the gradient of the logistic loss with L2 regularization
def gradient(beta, X, y, lambda_val):
    n = len(y)
    logits = X @ beta
    sigmoid = 1 / (1 + np.exp(-logits))
    grad = X.T @ (sigmoid - y) / n + lambda_val * beta
    return grad

# Gradient Descent optimization
costs = []
for t in range(num_iterations):
    cost = logistic_loss(beta, X, y, lambda_val)
    costs.append(cost)
    grad = gradient(beta, X, y, lambda_val)
    beta -= eta * grad
print (beta)
# Plot cost function versus iterations
plt.plot(range(num_iterations), costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations (Gradient Descent)')
plt.show()