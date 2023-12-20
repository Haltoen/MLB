import numpy as np
import matplotlib.pyplot as plt
import random

# Define the objective function and its gradient
def f(x):
    return np.exp(x[0] + 4 * x[1] - 0.3) + np.exp(x[0] - 4 * x[1] - 0.3) + np.exp(-x[0] - 0.3)

def R_gradient_f(x, case):
    if case == 1:
        df_dx1 = np.exp(x[0] + 4 * x[1] - 0.3) 
        df_dx2 = 4 * np.exp(x[0] + 4 * x[1] - 0.3) 

    elif case == 2:
        df_dx1 =  + np.exp(x[0] - 4 * x[1] - 0.3) 
        df_dx2 =  - 4 * np.exp(x[0] - 4 * x[1] - 0.3)
        
    elif case == 3:
        df_dx1 = np.exp(-x[0] - 0.3)
        df_dx2 = 0

  
    return np.array([df_dx1, df_dx2])

# Stochastic Gradient Descent algorithm with fixed step-size αt = 1 / t+1
def R_gradient_descent(starting_point, x_star, num_iterations, generations):
    all_f_values = []
    all_x_values = []
    for i in range(0, generations):
        x = starting_point
        f_values = []
        x_values = []

        for t in range(1, num_iterations + 1):
            case = random.randint(1,3)
            gradient = R_gradient_f(x, case)
            alpha = 1 / (t*10 + 1)
            x = x - alpha * gradient
            f_values.append(f(x) - f(x_star))
            x_values.append(np.linalg.norm(x - x_star))

        all_f_values.append(f_values)
        all_x_values.append(x_values)
    
    return all_f_values, all_x_values, x

# Initializations
starting_point = np.array([0, 0])
x_star = np.array([-0.5 * np.log(2), 0])
num_iterations = 100
num_descents = 25

# Run Gradient Descent
f_convergence, x_convergence, last_x = R_gradient_descent(starting_point, x_star, num_iterations, num_descents)
mean_f_convergence = np.mean(f_convergence, axis=0)
mean_x_convergence = np.mean(x_convergence, axis=0)

print(f"The average result of {num_descents} descenders after {num_iterations} iterations is (x,f(x))=({last_x},{f(last_x)})")

# Plot the convergence curves
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(mean_f_convergence, label='f(xt) - f(x*)')
plt.xlabel('Iterations')
plt.ylabel('Convergence Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mean_x_convergence, label='||xt - x*||_2')
plt.xlabel('Iterations')
plt.ylabel('Convergence Error')
plt.legend()

plt.tight_layout()
plt.show()