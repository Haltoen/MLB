import numpy as np
import matplotlib.pyplot as plt

# Load the data
X = np.loadtxt("X_data.csv", delimiter=',')
y = np.loadtxt("y_data.csv", delimiter=',')

# Set parameters
lambda_val = 5
eta = 1e-4  # Learning rate
num_iterations = 5000  # Number of iterations
batch_sizes = [1, 5, 20]  # Mini-batch sizes

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

# Function to calculate the gradient of the logistic loss with L2 regularization for a mini-batch
def gradient_mini_batch(beta, X, y, lambda_val, batch_indices):
    batch_X = X[batch_indices, :]
    batch_y = y[batch_indices]
    n = len(batch_y)
    logits = batch_X @ beta
    sigmoid = 1 / (1 + np.exp(-logits))
    grad = batch_X.T @ (sigmoid - batch_y) / n + lambda_val * beta
    return grad

# Stochastic Gradient Descent with different mini-batch sizes
for batch_size in batch_sizes:
    costs = []
    beta = np.zeros(d)  # Re-initialize beta with zeros for each batch size
    
    for t in range(num_iterations):
        indices = np.random.choice(len(y), batch_size, replace=False)
        cost = logistic_loss(beta, X, y, lambda_val)
        costs.append(cost)
        grad = gradient_mini_batch(beta, X, y, lambda_val, indices)
        beta -= eta * grad
    
    # Plot cost function versus iterations for each mini-batch size
    plt.plot(range(num_iterations), costs, label=f'Mini-batch Size: {batch_size}')

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function vs. Iterations (Stochastic Gradient Descent)')
plt.legend()
plt.show()