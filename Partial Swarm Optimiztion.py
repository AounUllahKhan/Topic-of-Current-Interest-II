import numpy as np
import math
import time
import matplotlib.pyplot as plt

# Control parameters
w = 0.5                   # Intertial weight.
c1 = 2.0                  # Cognitive parameter
c2 = 2.0                  # Social parameter
max_iter = 20
Np = 10                   # Population size
D = 2                     # Dimension

max_w = 0.9               # Maximum inertia weight
min_w = 0.4               # Minimum inertia weight

# Define constraints
x_lower = -5
x_upper = 5
y_lower = -1
y_upper = 20

xL = np.array([x_lower, y_lower])
xU = np.array([x_upper, y_upper])

def fitness(x):
    return (100 - x[0]**2) + (x[1]**2 - 56*x[0]*x[1]) - math.sin(x[0])

# Initialization
x = np.random.uniform(xL, xU, size=(Np, D))
v = np.random.rand(Np, D)

pbest = np.copy(x)
pbest_val = np.array([fitness(p) for p in pbest])
gbest = pbest[np.argmin(pbest_val)]
gbest_val = min(pbest_val)

gbest_store = np.zeros((max_iter, D))
pbest_val_avg_store = np.zeros(max_iter)
gbest_val_store = np.zeros(max_iter)

# PSO algorithm
t_start = time.time()

for iter in range(max_iter):
    # Update inertia weight
    w = max_w - (max_w - min_w) * iter / max_iter
    
    r1 = np.random.rand(Np, D)
    r2 = np.random.rand(Np, D)
        
    v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    
    # Clamp velocity
    v = np.clip(v, -1, 1)
    
    x = x + v
    
    # Clamp position
    x = np.clip(x, xL, xU)
    
    # Update pbest
    fit = np.array([fitness(p) for p in x])
    update_indices = fit < pbest_val
    pbest_val[update_indices] = fit[update_indices]
    pbest[update_indices] = np.copy(x[update_indices])
    
    # Update gbest
    gbest_index = np.argmin(pbest_val)
    if pbest_val[gbest_index] < gbest_val:
        gbest = np.copy(pbest[gbest_index])
        gbest_val = pbest_val[gbest_index]
    
    # Store gbest for plotting
    gbest_store[iter] = gbest
    pbest_val_avg_store[iter] = np.mean(pbest_val)
    gbest_val_store[iter] = gbest_val
    
    print("Iter. =", iter, ". gbest_val = ", gbest_val)

t_elapsed = time.time() - t_start
print("\nElapsed time = %.4f s" % t_elapsed)

# Plotting
plt.plot(pbest_val_avg_store, label='Average pbest value')
plt.plot(gbest_val_store, label='gbest value')
plt.xlabel('Iterations')
plt.ylabel('Fitness Value')
plt.legend()
plt.show()
