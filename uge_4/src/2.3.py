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

C_values = [1.0, 0.1, 0.01, 0.001]  # C values


free_sv_counts = []
bounded_sv_counts = []
non_sv_counts = []

for C in C_values:
    # Initialize and train the SVM model
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)

    # Get indices of support vectors
    support_vectors_indices = model.support_

    # Get dual coefficients (alphas) for support vectors
    alphas = np.abs(model.dual_coef_.ravel())

    # Calculate number of bounded, free, and non-support vectors
    num_data_points = len(X_train)
    num_bounded = np.sum(alphas[support_vectors_indices] == C) 
    num_free = np.sum((alphas[support_vectors_indices] > 0) & (alphas[support_vectors_indices] < C))  
    num_non_sv = num_data_points - (num_bounded + num_free)


    free_sv_counts.append(num_free)
    bounded_sv_counts.append(num_bounded)
    non_sv_counts.append(num_non_sv)
    total_sv_count = num_bounded + num_free + num_non_sv

    print(f"For C = {C}: Bounded SVs = {bounded_sv_counts}, Free SVs = {free_sv_counts}")
