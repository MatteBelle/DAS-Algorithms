import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
# Task 2.1
np.random.seed(0)

# Parameters
NN = 10  # Number of points

MAXITERS = 300
dd = 2    # Dimension of the input space
q = 4
gamma = 1
delta = 0
# Step 1: Generate a dataset
r = np.random.randn(NN, dd)
print(r)
print(r[:, 0])
def phi(z):
    return z

def grad_phi(z):
    return 1
# Step 2: Define the nonlinear transformation function phi
def sigma(z):
    return np.mean(phi(z), axis=0)

def cost_function(z, r, s):
    return gamma * np.linalg.norm(z - r)**2 + delta * np.linalg.norm(z - s)**2

def grad_function_1(z, r, s):
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1])

def grad_function_2(z, s):
    return 2 * delta * (z - s)

NN = 10

#G = nx.path_graph(NN)
G = nx.star_graph(NN-1)
#G = nx.cycle_graph(NN)

I_NN = np.eye(NN)

Adj = nx.adjacency_matrix(G)
Adj = Adj.toarray()
AA = np.zeros(shape=(NN, NN))

for ii in range(NN):
    N_ii = np.nonzero(Adj[ii])[0]
    deg_ii = len(N_ii)
    for jj in N_ii:
        deg_jj = len(np.nonzero(Adj[jj])[0])
        AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

AA += I_NN - np.diag(np.sum(AA, axis=0))

if 0:
    print(np.sum(AA, axis=0))
    print(np.sum(AA, axis=1))

#ZZ_at = np.random.randn(MAXITERS, NN, dd)

ZZ_at = np.zeros((MAXITERS, NN, dd))
SS_at = np.zeros((MAXITERS, NN, dd))
VV_at = np.zeros((MAXITERS, NN, dd))
for ii in range(NN):
    SS_at[0, ii] = phi(ZZ_at[0, ii])
    VV_at[0, ii] = grad_function_2(ZZ_at[0, ii], SS_at[0, ii])


cost_at = np.zeros((MAXITERS))
gradients_norm= np.zeros((MAXITERS))
gradients_k = np.zeros((q))
alpha = 1e-2

for kk in range(MAXITERS - 1):

    gradients_k = np.zeros((q))
    # gradient tracking
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]

        ZZ_at[kk + 1, ii] = ZZ_at[kk, ii] - alpha * (grad_function_1(ZZ_at[kk, ii], r[ii], SS_at[kk, ii]) + grad_phi(ZZ_at[kk, ii]) * VV_at[kk, ii])

        SS_at[kk + 1, ii] += AA[ii, ii] * SS_at[kk, ii]
        VV_at[kk + 1, ii] += AA[ii, ii] * VV_at[kk, ii]
        for jj in N_ii:
            VV_at[kk + 1, ii] += AA[ii, jj] * VV_at[kk, jj]
            SS_at[kk + 1, ii] += AA[ii, jj] * SS_at[kk, jj]

        SS_at[kk + 1, ii] += phi(ZZ_at[kk + 1, ii]) - phi(ZZ_at[kk, ii])

        new_grad_3 = grad_function_2(ZZ_at[kk + 1, ii], SS_at[kk + 1, ii])
        old_grad_3 = grad_function_2(ZZ_at[kk, ii], SS_at[kk, ii])
        VV_at[kk + 1, ii] += new_grad_3 - old_grad_3


        old_grad_1, old_grad_2 = grad_function_1(ZZ_at[kk, ii], r[ii], SS_at[kk, ii])
        new_grad_1, new_grad_2 = grad_function_1(ZZ_at[kk + 1, ii], r[ii], SS_at[kk, ii])
        gradients_k[0] += new_grad_1 - old_grad_1
        gradients_k[1] += new_grad_2 - old_grad_2

        gradients_k[2] += new_grad_3[0] - old_grad_3[0]
        gradients_k[3] += new_grad_3[1] - old_grad_3[1]

        ell_ii_gt = cost_function(ZZ_at[kk, ii], r[ii], SS_at[kk, ii])
        cost_at[kk] += ell_ii_gt

    gradients_norm[kk] = np.linalg.norm(gradients_k)

fig, ax = plt.subplots()
ax.semilogy(np.arange(MAXITERS - 1), cost_at[:-1])
ax.grid()

plt.show()

fig, ax = plt.subplots()
ax.semilogy(np.arange(MAXITERS - 1), gradients_norm[0:-1])
ax.grid()

plt.show()

def animation(ZZ_at, SS_at, NN, MAXITERS, r):
    color = ["r", "g", "b", "c", "m", "y", "#0072BD", "#D95319", "#7E2F8E", "#77AC30"]
    fig, ax = plt.subplots()
    for kk in range(MAXITERS):
        ax.cla()
        for ii in range(NN):
            if ii == 0:
                ax.plot(ZZ_at[kk, ii, 0], ZZ_at[kk, ii, 1], 'o', color=color[ii], label="Agents", markersize=15)
                ax.plot(r[ii, 0], r[ii, 1], 'x', color=color[ii], label="Targets", markersize=15)
            else:
                ax.plot(ZZ_at[kk, ii, 0], ZZ_at[kk, ii, 1], 'o', color=color[ii], markersize=15)
                ax.plot(r[ii, 0], r[ii, 1], 'x', color=color[ii], markersize=15)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        # also plot the baricenter
        ax.plot(SS_at[kk, 0, 0], SS_at[kk, 0, 1], 'p', color='k', label="Baricenter", markersize=15)
        ax.legend()
        # add legend 
        plt.xlabel("first component")
        plt.ylabel("second component")
        plt.title(f"Aggregative tracking - Simulation time = {kk}")
        plt.legend()
        plt.pause(0.1)
    plt.show()

animation(ZZ_at, SS_at, NN, MAXITERS, r)
