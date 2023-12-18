from scipy.stats import binom
from scipy.optimize import minimize_scalar



"""


max_prob = 0
best_p = 0

for p in range(0, 1001):
    p /= 1000
    prob_x = sum(binom.pmf(i, 10000, p) for i in range(9500, 10001))
    prob_y = p ** 100
    prob_simultaneous = prob_x * prob_y
    print(p, prob_x, prob_y, prob_simultaneous)
    if prob_simultaneous > max_prob:
        max_prob = prob_simultaneous
        best_p = p

print("Maximum probability:", max_prob)
print("Best value for p:", best_p)


from scipy.optimize import minimize_scalar

# Define the function to maximize
def simultaneous_probability(p):
    prob_x = sum(binom.pmf(i, 10000, p) for i in range(9500, 10001))
    prob_y = p ** 100
    return -prob_x * prob_y  # Negative to convert maximum to a minimum problem

# Find the p value that maximizes the simultaneous probability
result = minimize_scalar(simultaneous_probability, bounds=(0, 1), method='bounded')
best_p = result.x
max_prob = -result.fun  # Convert back to maximum probability

print("Maximum probability:", max_prob)
print("Best value for p:", best_p)
"""

"""
from scipy.stats import binom

max_probability = 0
worst_p = 0

# Iterate through various p values from 0 to 1
for p_value in range(0, 1001):
    p = p_value / 1000  # Convert to a float between 0 and 1
    # Calculate the probabilities for both events
    event1_prob = binom.cdf(9499, 10000, p)  # Probability of 9499 or fewer show-ups in 10,000 reservations
    event2_prob = p ** 100  # Probability that all 100 passengers show up
    
    # Calculate the simultaneous probability of both events
    simultaneous_prob = event1_prob * event2_prob
    
    # Update maximum probability and worst-case p if a higher probability is found
    if simultaneous_prob > max_probability:
        max_probability = simultaneous_prob
        worst_p = p

print("Maximum probability of simultaneous events:", max_probability)
print("Worst-case p:", worst_p)



# Calculate the worst-case p
p_worst_case = 0.95 ** 100

# Calculate the probability of observing both events simultaneously
probability_bound = 0.95 ** 10000 * p_worst_case ** 100

print("Worst-case p:", p_worst_case)
print("Probability bound:", probability_bound)"""

"""
from math import comb, log

# Function to calculate the logarithm of Event A probability
def log_event_a_probability(p):
    return (log(comb(10000, 9500)) + (9500) * log(p) + (500) * log(1 - p))

# Function to calculate the logarithm of Event B probability
def log_event_b_probability(p):
    return 100 * log(p)

# Define convergence threshold
threshold = 1e-3  # Adjust if needed

# Initial guess for p
p = 0.95

# Iterative computation to find the maximum p
while True:
    print(p)
    log_prob_a = log_event_a_probability(p)
    log_prob_b = log_event_b_probability(p)
    
    # Check convergence
    if abs(log_prob_a - log_prob_b) < log(1 + threshold):
        break
    
    # Adjust p based on the comparison of logarithmic probabilities
    if log_prob_a > log_prob_b:
        p -= 0.000001  # Decrease p
    else:
        p += 0.000001  # Increase p

    # Ensure p stays within [0, 1]
    p = max(min(p, 1), 0)

# Output the maximum p satisfying both conditions
print("Maximum p satisfying both conditions:", p)
"""
import math
from scipy.optimize import minimize

def kl_divergence(p, q):
    if p in (0, 1) or q in (0, 1):
        return 0  # Return 0 when p or q is 0 or 1
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

# Function to calculate the upper bound using KL inequality
def kl_upper_bound(p):
    # Event 1: 95% show-ups in a sample of 10000 passengers
    event_1 = kl_divergence(p, 0.95)
    
    # Event 2: 100% show-ups in a sample of 100 passengers
    event_2 = kl_divergence(p, 1)
    
    # Calculate the joint probability upper bound using KL inequality
    joint_probability_bound = math.exp(event_1 + event_2)
    
    return joint_probability_bound

# Find the worst-case p by minimizing the negative of the joint probability bound
result = minimize(lambda p: -kl_upper_bound(p), x0=0.5, bounds=[(0.0001, 0.9999)])

# Maximum value of the joint probability bound
max_joint_probability_bound = kl_upper_bound(result.x[0])

print(f"The worst-case probability bound is approximately: {max_joint_probability_bound:.6f}")