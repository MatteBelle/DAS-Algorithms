# Task 2.3: Aggregative tracking with walls
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

# Parameters
np.random.seed(0)
NN = 10  # Number of agents
MAXITERS = 300
dd = 2      # Dimension of the input space
q = 4       # Dimension of the output space
gamma = 0.5 # Weight of the targets
delta = 0.1 # Weight of the baricenter
epsilon = 2 # Weight of the walls
r = np.random.randn(NN, dd) # Targets (random)
for n in range(NN):
    r[n] = r[n] + (8,0) # Each target is shifted by (8,0) to the right
# Walls definition
wall1 = np.zeros((2,2))
wall2 = np.zeros((2,2))
# Walls are defined only by two points
wall1[0] = (-4,1)
wall1[1] = (4,1)
wall2[0] = (-4,-1)
wall2[1] = (4,-1)

# Functions definition
def phi(z):
    '''
    Nonlinear transformation function phi
    In this case, phi is the identity function
    Input:
    z: input vector
    Output:
    z: output vector
    '''
    return z

def grad_phi(z):
    '''
    Gradient of the nonlinear transformation function phi
    Input:
    z: input vector
    Output:
    1: gradient of the output vector
    '''
    return 1

def sigma(z):
    '''
    Nonlinear transformation function sigma
    Input:
    z: input vector
    Output:
    mean: mean of phi(z)
    '''
    return np.mean(phi(z), axis=0)

def cost_function(z, r, s, w1, w2):
    '''
    Cost function
    Input:
    z: input vector
    r: target vector
    s: baricenter vector
    w1: first wall
    w2: second wall
    Output:
    cost: cost value
    '''
    return gamma * np.linalg.norm(z - r)**2 + delta * np.linalg.norm(z - s)**2 + epsilon * (np.linalg.norm(z[1] - (w1[0][1] + w2[0][1]) / 2)**2)

def grad_function_1(z, r, s, w1, w2):
    '''
    Gradient of the cost function
    Input:
    z: input vector
    r: target vector
    s: baricenter vector
    w1: first wall
    w2: second wall
    Output:
    grad_1: gradient of the cost function of the first component
    grad_2: gradient of the cost function of the second component
    '''
    c = 0
    if (z[0] < w1[1][0]):
        c = 2 * epsilon * (z[1] - (w1[0][1] + w2[0][1]) / 2)
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1]) + c

def grad_function_2(z, s):
    '''
    Gradient of the cost function (baricenter)
    Input:
    z: input vector
    s: baricenter vector
    Output:
    grad: gradient of the cost function
    '''
    return 2 * delta * (z - s)

# Graph definition
#G = nx.path_graph(NN)
#G = nx.star_graph(NN-1)
G = nx.cycle_graph(NN)

I_NN = np.eye(NN)
Adj = nx.adjacency_matrix(G)
Adj = Adj.toarray()
AA = np.zeros(shape=(NN, NN))

# Initialization of the AA matrix
for ii in range(NN):
    N_ii = np.nonzero(Adj[ii])[0]
    deg_ii = len(N_ii)
    for jj in N_ii:
        deg_jj = len(np.nonzero(Adj[jj])[0])
        AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

AA += I_NN - np.diag(np.sum(AA, axis=0))

# Initialization of the variables
ZZ_at = np.random.randn(MAXITERS, NN, dd) # Agents
for n in range(NN):
    ZZ_at[0, n] = ZZ_at[0, n] - (10,0) # Each agent is shifted by (10,0) to the left
SS_at = np.zeros((MAXITERS, NN, dd)) # Baricenter 
VV_at = np.zeros((MAXITERS, NN, dd)) # Gradient of the cost function
for ii in range(NN):
    SS_at[0, ii] = phi(ZZ_at[0, ii])
    VV_at[0, ii] = grad_function_2(ZZ_at[0, ii], SS_at[0, ii])

# Gradient descent
cost_at = np.zeros((MAXITERS))
gradients_norm = np.zeros((MAXITERS))
alpha = 1e-2 # Step size

for kk in range(MAXITERS - 1):
    
    gradients_k = np.zeros((q))
    # gradient tracking
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]

        ZZ_at[kk + 1, ii] = ZZ_at[kk, ii] - alpha * (grad_function_1(ZZ_at[kk, ii], r[ii], SS_at[kk, ii], wall1, wall2) + grad_phi(ZZ_at[kk, ii]) * VV_at[kk, ii])

        SS_at[kk + 1, ii] += AA[ii, ii] * SS_at[kk, ii]
        VV_at[kk + 1, ii] += AA[ii, ii] * VV_at[kk, ii]
        for jj in N_ii:
            VV_at[kk + 1, ii] += AA[ii, jj] * VV_at[kk, jj]
            SS_at[kk + 1, ii] += AA[ii, jj] * SS_at[kk, jj]

        SS_at[kk + 1, ii] += phi(ZZ_at[kk + 1, ii]) - phi(ZZ_at[kk, ii])

        new_grad_3 = grad_function_2(ZZ_at[kk + 1, ii], SS_at[kk + 1, ii])
        old_grad_3 = grad_function_2(ZZ_at[kk, ii], SS_at[kk, ii])
        VV_at[kk + 1, ii] += new_grad_3 - old_grad_3

        old_grad_1, old_grad_2 = grad_function_1(ZZ_at[kk, ii], r[ii], SS_at[kk, ii], wall1, wall2)
        new_grad_1, new_grad_2 = grad_function_1(ZZ_at[kk + 1, ii], r[ii], SS_at[kk, ii], wall1, wall2)
        gradients_k[0] = old_grad_1
        gradients_k[1] = old_grad_2
        gradients_k[2] = old_grad_3[0]
        gradients_k[3] = old_grad_3[1]

        ell_ii_gt = cost_function(ZZ_at[kk, ii], r[ii], SS_at[kk, ii], wall1, wall2)
        cost_at[kk] += ell_ii_gt
        gradients_norm[kk] += np.linalg.norm(gradients_k)

# Plots: cost function and gradient norm
fig, ax = plt.subplots()
ax.semilogy(np.arange(MAXITERS - 2), cost_at[1:-1])
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.title("Evolution of the cost function")
ax.grid()
plt.show()

fig, ax = plt.subplots()
ax.semilogy(np.arange(MAXITERS - 2), gradients_norm[1:-1])
plt.xlabel("Iterations")
plt.ylabel("Gradient norm")
plt.title("Evolution of the gradient norm")
ax.grid()
plt.show()

# Animation of the agents and the targets
def animation(ZZ_at, SS_at, NN, MAXITERS, r):
    color = ["r", "g", "b", "c", "m", "y", "#0072BD", "#D95319", "#7E2F8E", "#77AC30"]
    fig, ax = plt.subplots()
    for kk in range(MAXITERS):
        ax.cla()
        for ii in range(NN):
            if ii == 0:
                ax.plot(ZZ_at[kk, ii, 0], ZZ_at[kk, ii, 1], 'o', color=color[ii], label="Agents", markersize=5)
                ax.plot(r[ii, 0], r[ii, 1], 'x', color=color[ii], label="Targets", markersize=10)
            else:
                ax.plot(ZZ_at[kk, ii, 0], ZZ_at[kk, ii, 1], 'o', color=color[ii], markersize=5)
                ax.plot(r[ii, 0], r[ii, 1], 'x', color=color[ii], markersize=10)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-10, 10)
        # plot the baricenter
        ax.plot(SS_at[kk, 0, 0], SS_at[kk, 0, 1], 'p', color='k', label="Baricenter", markersize=5)
        # plot the walls
        ax.plot(wall1[:,0], wall1[:,1], 'k', label="Wall", color="green", linestyle="solid")
        ax.plot(wall2[:,0], wall2[:,1], 'k', color="green", linestyle="solid")
        ax.legend()
        plt.xlabel("first component")
        plt.ylabel("second component")
        plt.title(f"Aggregative tracking - Simulation time = {kk}")
        plt.legend()
        plt.pause(0.1)
    plt.show()

animation(ZZ_at, SS_at, NN, MAXITERS, r)