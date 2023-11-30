import numpy as np
import matplotlib.pyplot as plt
delta = 0.01
n = 1000
p_hat_values = np.arange(0.000, 1.001, 0.001)  # Generate p_n in {0, 0.001, ..., 1}

print(p_hat_values)


def kl_divergence(p, q):
    epsilon = 1e-10  # Small value to avoid zero or near-zero inputs
    p = np.clip(p, epsilon, 1 - epsilon)  # Ensure p is not exactly 0 or 1
    q = np.clip(q, epsilon, 1 - epsilon)  # Ensure q is not exactly 0 or 1
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))



# Define the bounds functions
def hoeffding_bound(p_hat, delta, n):
    return p_hat + np.sqrt(np.log(1 / delta) / (2 * n))

def binary_search(p_hat, z):
    upper =1
    lower = p_hat
    current_p = round((lower + upper)/2, 3)
    print ("z",z)
    
    while True:
        bound = kl_divergence(p_hat,current_p)
        print ("bound",bound)
        print("u/l", upper, lower)
        if bound >= z:
            upper = current_p - 0.001
        else:
            lower = current_p

        new_p = round((lower + upper)/2, 3)
        if current_p == new_p:
            break
        current_p = new_p # update current_p

    return current_p 

def kl_inverse_bound(p_hat, delta, n):
    z = np.log((n + 1) / delta) / n

    max_p_values = np.zeros_like(p_hat)  # Initialize an array to store max_p values
    for i, p in enumerate(p_hat):
        max_p_values[i] = binary_search(p, z)  # Perform binary search for each p_hat element
    
    return max_p_values



def pinsker_bound(p_hat, delta, n):
    return p_hat + np.sqrt(np.log((n + 1) / delta) / (2 * n))

def refined_pinsker_bound(p_hat, delta, n):
    return p_hat + np.sqrt(2 * p_hat * np.log((n + 1) / delta) / (2 * n)) + 2 * np.log((n + 1) / delta) / n

# Parameters


# Calculate bounds
hoeffding_bounds = hoeffding_bound(p_hat_values, delta, n)
kl_inverse_bounds = kl_inverse_bound(p_hat_values, delta, n)
pinsker_bounds = pinsker_bound(p_hat_values, delta, n)
refined_pinsker_bounds = refined_pinsker_bound(p_hat_values, delta, n)

# Clip all bounds at 1
hoeffding_bounds = np.clip(hoeffding_bounds, None, 1)
kl_inverse_bounds = np.clip(kl_inverse_bounds, None, 1)
pinsker_bounds = np.clip(pinsker_bounds, None, 1)
refined_pinsker_bounds = np.clip(refined_pinsker_bounds, None, 1)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Full range plot
ax1.plot(p_hat_values, hoeffding_bounds, label="Hoeffding's Inequality")
ax1.plot(p_hat_values, kl_inverse_bounds, label="KL Inequality (Upper Bound)")
ax1.plot(p_hat_values, pinsker_bounds, label="Pinsker's Relaxation")
ax1.plot(p_hat_values, refined_pinsker_bounds, label="Refined Pinsker's Relaxation")
ax1.set_xlabel(r'$\hat{p}_n$')
ax1.set_ylabel('$p$')
ax1.set_title('Bounds on $p$ as a function of $\hat{p}_n$')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, 1.1)  # Limit y-axis to [0, 1] as suggested

# Zoomed-in plot
ax2.plot(p_hat_values, hoeffding_bounds, label="Hoeffding's Inequality")
ax2.plot(p_hat_values, kl_inverse_bounds, label="KL Inequality (Upper Bound)")
ax2.plot(p_hat_values, pinsker_bounds, label="Pinsker's Relaxation")
ax2.plot(p_hat_values, refined_pinsker_bounds, label="Refined Pinsker's Relaxation")
ax2.set_xlabel(r'$\hat{p}_n$')
ax2.set_ylabel('$p$')
ax2.set_title('Zoomed-in: Bounds on $p$ as a function of $\hat{p}_n$ (0 to 0.1)')
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 0.1)  # Limit x-axis to [0, 0.1]
ax2.set_ylim(0, 0.2)  # Limit y-axis to [0, 1] as suggested

plt.tight_layout()
plt.show()