# from __future__ import print_function    

# import time
# import math
# import numpy as np
# import pylab as py
# py.rcParams.update({'font.size': 14})


# # Control parameters
# w = 0.5                   # Intertial weight. In some variations, it is set to vary with iteration number.
# c1 = 2.0                  # Weight of searching based on the optima found by a particle
# c2 = 2.0                  # Weight of searching based on the optima found by the swarm
# v_fct = 1                 # Velocity adjust factor. Set to 1 for standard PSO.

# max_iter=50
# Np = 50                   # population size (number of particles)
# D = 2   
#                   # dimension (= no. of parameters in the fitness function)
# max_w = 0.9   # Maximum inertia weight
# min_w = 0.4   # Minimum inertia weight

# w = max_w     # Initial inertia weight         # maximum number of iterations 

# # Define constraints
# x_lower = -5
# x_upper = 5
# y_lower = -1
# y_upper = 20

# xL = np.array([x_lower, y_lower])     # lower bound (does not need to be homogeneous)  
# xU = np.array([x_upper, y_upper])     # upper bound (does not need to be homogeneous)   



# def fitness(x):
#     # x is a matrix of size D x Np
#     # The position of the entire swarm is inputted at once. 
#     # Thus, one function call evaluates the fitness value of the entire swarm
#     # F is a vector of size Np. Each element represents the fitness value of each particle in the swarm
    
#     F_sphere = np.zeros(x.shape[1])  # Initialize array to store fitness values for each particle
    
#     for i in range(x.shape[1]):  # Loop through each particle
#         F_sphere[i] = (100 - x[0,i]**2) + (x[1,i]**2 - 56*x[0,i]*x[1,i]) - (math.sin(x[0,i]))
    
#     return F_sphere


# # Defining and initializing variables
    
# pbest_val = np.zeros(Np)            # Personal best fitness value. One pbest value per particle.
# gbest_val = np.zeros(max_iter)      # Global best fitness value. One gbest value per iteration (stored).

# pbest = np.zeros((D,Np))            # pbest solution
# gbest = np.zeros(D)                 # gbest solution

# gbest_store = np.zeros((D,max_iter))   # storing gbest solution at each iteration

# pbest_val_avg_store = np.zeros(max_iter)
# fitness_avg_store = np.zeros(max_iter)

# x = np.random.rand(D,Np)            # Initial position of the particles
# v = np.zeros((D,Np))                # Initial velocity of the particles




# # Setting the initial position of the particles over the given bounds [xL,xU]
# for m in range(D):    
#     x[m,:] = xL[m] + (xU[m]-xL[m])*x[m,:]
    



# t_start = time.time()

# # Loop over the generations
# for iter in range(0,max_iter):
#     # Update inertia weight
#     w = max_w - (max_w - min_w) * iter / max_iter                         # Do not update position for 0th iteration
#     r1 = np.random.rand(D,Np)            # random numbers [0,1], matrix D x Np
#     r2 = np.random.rand(D,Np)            # random numbers [0,1], matrix D x Np   
        
#     v = v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest.reshape(-1,1) - x)  # Velocity update
        
#     x = x + v                            # Position update                     # position update
    
    
#     fit = fitness(x)                         # fitness function call (once per iteration). Vector Np
    
#     if iter == 0:
#         pbest_val = np.copy(fit)             # initial personal best = initial fitness values. Vector of size Np
#         pbest = np.copy(x)                   # initial pbest solution = initial position. Matrix of size D x Np
#     else:
#         # pbest and pbest_val update
#         ind = np.argwhere(fit > pbest_val)   # indices where current fitness value set is greater than pbset
#         pbest_val[ind] = np.copy(fit[ind])   # update pbset_val at those particle indices where fit > pbest_val
#         pbest[:,ind] = np.copy(x[:,ind])     # update pbest for those particle indices where fit > pbest_val
      
#     # gbest and gbest_val update
#     ind2 = np.argmax(pbest_val)                       # index where the fitness is maximum
#     gbest_val[iter] = np.copy(pbest_val[ind2])        # store gbest value at each iteration
#     gbest = np.copy(pbest[:,ind2])                    # global best solution, gbest
    
#     gbest_store[:,iter] = np.copy(gbest)              # store gbest solution
    
#     pbest_val_avg_store[iter] = np.mean(pbest_val)
#     fitness_avg_store[iter] = np.mean(fit)
#     print("Iter. =", iter, ". gbest_val = ", gbest_val[iter])  # print iteration no. and best solution at each iteration
    
    

# t_elapsed = time.time() - t_start
# print("\nElapsed time = %.4f s" % t_elapsed)



# # Plotting
# py.close('all')
# py.plot(gbest_val,label = 'gbest_val')
# py.legend()
# py.xlabel('iterations')
# py.ylabel('gbest_val')
# py.show()

# # py.figure()
# # for m in range(D):
# #     py.plot(gbest_store[m,:],label = 'D = ' + str(m+1))
# # py.legend()
# # py.xlabel('iterations')
# # py.ylabel('Best solution, gbest[:,iter]')
# # py.show()

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
