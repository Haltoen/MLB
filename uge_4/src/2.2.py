from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import zero_one_loss
import numpy as np

y_test = np.loadtxt('y_test_binary.csv', delimiter = ',' )
y_train = np.loadtxt('y_train_binary.csv', delimiter=',')
X_test = np.loadtxt ('X_test_binary.csv', delimiter = ',')
X_train = np.loadtxt ('X_train_binary.csv', delimiter = ',')

# Normalizing the input data
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)

def f_norm (input): # based on x_train
    normalized = (input - train_mean) / train_std
    return normalized

# trainsform input data
X_train = f_norm(X_train)
X_test = f_norm(X_test) 

"""
# Shuffle Training data
random_seed = 7
np.random.seed(random_seed)
num_samples = X_train.shape[0]  # Number of samples
shuffled_indices = np.random.permutation(num_samples)
X_train = X_train[shuffled_indices]
"""


# Define the grid of hyperparameters
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}


# Initialize SVM classifier
svm = SVC(kernel='rbf')

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=5, verbose = 0, n_jobs = -1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Train Best sv
best_svm = SVC( kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
best_svm.fit(X_train, y_train)

# Evaluate performance on training data
train_predictions = best_svm.predict(X_train)
train_loss = zero_one_loss(y_train, train_predictions)

# Evaluate performance on test data
test_predictions = best_svm.predict(X_test)
test_loss = zero_one_loss(y_test, test_predictions)

# Report results
print("Best hyperparameters (C, gamma):", best_params)
print("0-1 Loss on training data:", train_loss)
print("0-1 Loss on test data:", test_loss)