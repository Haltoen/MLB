import numpy as np
import matplotlib.pyplot as plt

y_test = np.loadtxt('y_test_binary.csv', delimiter = ',' )
y_train = np.loadtxt('y_train_binary.csv', delimiter=',')
X_test = np.loadtxt ('X_test_binary.csv', delimiter = ',')
X_train = np.loadtxt ('X_train_binary.csv', delimiter = ',')

# Get sample counts
train_samples = len(y_train)
test_samples = len(y_test)
print (f"Amount of samples \n Train: {train_samples}  Test : {test_samples}")


# Frequency of classes with plot
train_uniques, train_counts = np.unique(y_train, return_counts=True)
test_uniques, test_counts = np.unique(y_test, return_counts=True)

train_freq = train_counts / sum(train_counts)
test_freq = test_counts/ sum(test_counts)
bar_width = 1.75

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.bar(train_uniques, train_freq, width=bar_width, label="Training distribution") 
ax1.set_xticks([-1, 1])
ax1.set_xlim(-2, 2) 
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)

ax2.bar(test_uniques, test_freq, width=bar_width, label="Test distribution") 
ax2.set_xticks([-1, 1])
ax2.set_xlim(-2, 2) 
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1)
    

# Normalizing the input data
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)

def f_norm (input): # based on x_train
    normalized = (input - train_mean) / train_std
    return normalized

norm_x_train = f_norm(X_train)
norm_x_test = f_norm(X_test) 

print (f"normilzed x_train stats \n Mean: {np.mean(norm_x_train)} \n Variance: {np.var(norm_x_train)}")
print (f"normilzed y_train stats \n Mean: {np.mean(norm_x_test)} \n Variance: {np.var(norm_x_test)}")

plt.show() # plot last