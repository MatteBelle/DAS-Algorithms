# Task 2.1: Aggregative tracking - Square target with 4 points (moving targets)
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

# Parameters definition
np.random.seed(0)
NN = 4  # Number of agents
MAXITERS = 200
dd = 2       # Dimension of the input space
q = 4        # Dimension of the output space
gamma = 0.25 # Weight of the targets
delta = 0.75 # Weight of the baricenter
# Step 1: Generate a dataset
r = np.zeros((MAXITERS, NN, dd)) # Targets (square)
r[0] = np.array([[1,1], [1,-1], [-1,-1], [-1,1]]) # Initial square position

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

def cost_function(z, r, s):
    '''
    Cost function 
    Input:
    z: input vector
    r: target vector
    s: baricenter vector
    Output:
    cost: cost value
    '''
    return gamma * np.linalg.norm(z - r)**2 + delta * np.linalg.norm(z - s)**2

def grad_function_1(z, r, s):
    '''
    Gradient of the cost function
    Input:
    z: input vector
    r: target vector
    s: baricenter vector
    Output:
    grad_1: gradient of the cost function of the first component
    grad_2: gradient of the cost function of the second component
    '''
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1])

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
G = nx.star_graph(NN-1)
#G = nx.cycle_graph(NN)

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

# Initialization of the variables: Metropolis-Hastings weights
#ZZ_at = np.random.randn(MAXITERS, NN, dd)
ZZ_at = np.zeros((MAXITERS, NN, dd)) # Agents
SS_at = np.zeros((MAXITERS, NN, dd)) # Baricenter
VV_at = np.zeros((MAXITERS, NN, dd)) # Gradient of the cost function
for ii in range(NN):
    SS_at[0, ii] = phi(ZZ_at[0, ii])
    VV_at[0, ii] = grad_function_2(ZZ_at[0, ii], SS_at[0, ii])

# Gradient descent
cost_at = np.zeros((MAXITERS))
gradients_norm_z= np.zeros((MAXITERS))
gradients_norm_s = np.zeros((MAXITERS))
gradients_k = np.zeros((q))
alpha = 1e-2 # Step size

for kk in range(MAXITERS - 1):

    gradients_k = np.zeros((q))
    # gradient tracking
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]

        ZZ_at[kk + 1, ii] = ZZ_at[kk, ii] - alpha * (grad_function_1(ZZ_at[kk, ii], r[kk, ii], SS_at[kk, ii]) + grad_phi(ZZ_at[kk, ii]) * VV_at[kk, ii])

        SS_at[kk + 1, ii] += AA[ii, ii] * SS_at[kk, ii]
        VV_at[kk + 1, ii] += AA[ii, ii] * VV_at[kk, ii]
        for jj in N_ii:
            VV_at[kk + 1, ii] += AA[ii, jj] * VV_at[kk, jj]
            SS_at[kk + 1, ii] += AA[ii, jj] * SS_at[kk, jj]

        SS_at[kk + 1, ii] += phi(ZZ_at[kk + 1, ii]) - phi(ZZ_at[kk, ii])

        new_grad_3 = grad_function_2(ZZ_at[kk + 1, ii], SS_at[kk + 1, ii])
        old_grad_3 = grad_function_2(ZZ_at[kk, ii], SS_at[kk, ii])
        VV_at[kk + 1, ii] += new_grad_3 - old_grad_3


        old_grad_1, old_grad_2 = grad_function_1(ZZ_at[kk, ii], r[kk, ii], SS_at[kk, ii])
        new_grad_1, new_grad_2 = grad_function_1(ZZ_at[kk + 1, ii], r[kk, ii], SS_at[kk, ii])
        gradients_k[0] = old_grad_1
        gradients_k[1] = old_grad_2
        gradients_k[2] = old_grad_3[0]
        gradients_k[3] = old_grad_3[1]

        ell_ii_gt = cost_function(ZZ_at[kk, ii], r[kk, ii], SS_at[kk, ii])
        cost_at[kk] += ell_ii_gt
        gradients_norm_z[kk] += np.linalg.norm(gradients_k[:2])
        gradients_norm_s[kk] += np.linalg.norm(gradients_k[2:])

    # Update the targets position
    r[kk + 1] = np.array([r[kk,0] - [0.005,0.005], r[kk,1] - [0.005,-0.005], r[kk,2] - [-0.005,-0.005], r[kk,3] - [-0.005,0.005]])

# Plots: cost function and gradient norm
fig, ax = plt.subplots()
ax.semilogy(np.arange(MAXITERS - 2), cost_at[1:-1])
plt.xlabel("Iterations")
plt.ylabel("Cost function")
plt.title("Evolution of the cost function")
ax.grid()
plt.show()

fig, ax = plt.subplots()
ax.semilogy(np.arange(MAXITERS - 2), gradients_norm_z[1:-1])
plt.xlabel("Iterations")
plt.ylabel("Gradient norm of ZZ")
plt.title("Evolution of the gradient norm of ZZ")
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
                ax.plot(ZZ_at[kk, ii, 0], ZZ_at[kk, ii, 1], 'o', color=color[ii], label="Agents", markersize=15)
                ax.plot(r[kk, ii, 0], r[kk, ii, 1], 'x', color=color[ii], label="Targets", markersize=15)
            else:
                ax.plot(ZZ_at[kk, ii, 0], ZZ_at[kk, ii, 1], 'o', color=color[ii], markersize=15)
                ax.plot(r[kk, ii, 0], r[kk, ii, 1], 'x', color=color[ii], markersize=15)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        # plot the baricenter
        ax.plot(SS_at[kk, 0, 0], SS_at[kk, 0, 1], 'p', color='k', label="Baricenter", markersize=15)
        ax.legend()
        plt.xlabel("first component")
        plt.ylabel("second component")
        plt.title(f"Aggregative tracking - Simulation time = {kk}")
        plt.legend()
        plt.pause(0.1)
    plt.show()

animation(ZZ_at, SS_at, NN, MAXITERS, r)
