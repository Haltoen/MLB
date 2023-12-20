import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and its gradient
def f(x):
    return np.exp(x[0] + 4 * x[1] - 0.3) + np.exp(x[0] - 4 * x[1] - 0.3) + np.exp(-x[0] - 0.3)

def gradient_f(x):
    df_dx1 = np.exp(x[0] + 4 * x[1] - 0.3) + np.exp(x[0] - 4 * x[1] - 0.3) - np.exp(-x[0] - 0.3)
    df_dx2 = 4 * np.exp(x[0] + 4 * x[1] - 0.3) - 4 * np.exp(x[0] - 4 * x[1] - 0.3)
    return np.array([df_dx1, df_dx2])

# Gradient Descent algorithm with fixed step-size αt = 1 / t+1
def gradient_descent(starting_point, x_star, num_iterations):
    x = starting_point
    f_values = []
    x_values = []

    for t in range(1, num_iterations + 1):
        gradient = gradient_f(x)
        alpha = 1 / (t + 1)
        x = x - alpha * gradient
        f_values.append(f(x) - f(x_star))
        x_values.append(np.linalg.norm(x - x_star))

    return f_values, x_values, x

# Initializations
starting_point = np.array([0, 0])
x_star = np.array([-0.5 * np.log(2), 0])
num_iterations = 10

# Run Gradient Descent
f_convergence, x_convergence, last_x = gradient_descent(starting_point, x_star, num_iterations)

print(f" After {num_iterations} iterations the result of GD is: (x,f(x))=({last_x},{f(last_x)})")

# Plot the convergence curves
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(f_convergence, label='f(xt) - f(x*)')
plt.xlabel('Iterations')
plt.ylabel('Convergence Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_convergence, label='||xt - x*||_2')
plt.xlabel('Iterations')
plt.ylabel('Convergence Error')
plt.legend()

plt.tight_layout()
plt.show()