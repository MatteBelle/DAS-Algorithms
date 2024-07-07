import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
# Task 2.1
np.random.seed(0)
# Parameters
NN = 10  # Number of points
d = 2    # Dimension of the input space
q = 4
gamma = 0.5
delta = 0.1
epsilon = 2
# Step 1: Generate a dataset
r = np.random.randn(NN, d)
for n in range(NN):
    r[n] = r[n] + (8,0)

wall1 = np.zeros((2,2))
wall2 = np.zeros((2,2))

wall1[0] = (-4,1)
wall1[1] = (4,1)
wall2[0] = (-4,-1)
wall2[1] = (4,-1)

def phi(z):
    return z

def grad_phi(z):
    return 1
# Step 2: Define the nonlinear transformation function phi
def sigma(z):
    return np.mean(phi(z), axis=0)
def cost_function(z, r, s, w1, w2):
    return gamma * np.linalg.norm(z - r)**2 + delta * np.linalg.norm(z - s)**2 + epsilon * (np.linalg.norm(z[1] - (w1[0][1] + w2[0][1]) / 2)**2)
def grad_function_1(z, r, s, w1, w2):
    c = 0
    if (z[0] < w1[1][0]):
        c = 2 * epsilon * (z[1] - (w1[0][1] + w2[0][1]) / 2)
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1]) + c

def grad_function_2(z, s):
    return 2 * delta * (z - s)


NN = 10

#G = nx.path_graph(NN)
#G = nx.star_graph(NN-1)
G = nx.cycle_graph(NN)

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
MAXITERS = 1000
dd = 2

ZZ_at = np.random.randn(MAXITERS, NN, dd)
for n in range(NN):
    ZZ_at[0, n] = ZZ_at[0, n] - (15,0)
SS_at = np.zeros((MAXITERS, NN, dd))
VV_at = np.zeros((MAXITERS, NN, dd))
for ii in range(NN):
    SS_at[0, ii] = phi(ZZ_at[0, ii])
    VV_at[0, ii] = grad_function_2(ZZ_at[0, ii], SS_at[0, ii])


cost_at = np.zeros((MAXITERS))
gradients_norm = np.zeros((MAXITERS))
alpha = 1e-2
for kk in range(MAXITERS - 1):

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

        new_grad = grad_function_2(ZZ_at[kk + 1, ii], SS_at[kk + 1, ii])
        old_grad = grad_function_2(ZZ_at[kk, ii], SS_at[kk, ii])
        VV_at[kk + 1, ii] += new_grad - old_grad

        gradient_norm = np.linalg.norm(new_grad - old_grad)
        gradients_norm[kk] = gradient_norm

        ell_ii_gt = cost_function(ZZ_at[kk, ii], r[ii], SS_at[kk, ii], wall1, wall2)
        cost_at[kk] += ell_ii_gt


fig, ax = plt.subplots()
ax.plot(np.arange(MAXITERS - 1), cost_at[:-1])
ax.grid()

plt.show()

fig, ax = plt.subplots()
ax.plot(np.arange(MAXITERS - 1), gradients_norm[0:-1])
ax.grid()

plt.show()
def animation(ZZ_at, NN, MAXITERS, r):

    '''
    for tt in range(len(horizon)):
        # plot trajectories
        plt.plot(
            XX[0 : n_x * NN : n_x].T,
            XX[1 : n_x * NN : n_x].T,
            color=gray_O4S,
            linestyle="dashed",
            alpha=0.5,
        )

        # plot formation
        xx_tt = XX[:, tt].T
        for ii in range(NN):
            index_ii = ii * n_x + np.arange(n_x)
            p_prev = xx_tt[index_ii]

            plt.plot(
                p_prev[0],
                p_prev[1],
                marker="o",
                markersize=15,
                fillstyle="full",
                color=red_O4S,
            )

            for jj in range(NN):
                if Adj[ii, jj] & (jj > ii):
                    index_jj = (jj % NN) * n_x + np.arange(n_x)
                    p_curr = xx_tt[index_jj]
                    plt.plot(
                        [p_prev[0], p_curr[0]],
                        [p_prev[1], p_curr[1]],
                        linewidth=1,
                        color=emph_O4S,
                        linestyle="solid",
                    )
        '''


    for tt in range(MAXITERS):
        plt.plot(
            r[:, 0],
            r[:, 1],
            marker="x",
            markersize=15,
            color="red",
        )
        plt.plot(
            wall1[:,0],
            wall1[:,1],
            linewidth=1,
            color="green",
            linestyle="solid",
        )
        plt.plot(
            wall2[:,0],
            wall2[:,1],
            linewidth=1,
            color="green",
            linestyle="solid",
        )
        plt.plot(
            ZZ_at[tt, :, 0],
            ZZ_at[tt, :, 1],
            marker="o",
            markersize=15,
            color="blue",
        )
        axes_lim = (-10, 10)
        plt.xlim(axes_lim)
        plt.ylim(axes_lim)
        plt.axis("equal")
        plt.xlabel("first component")
        plt.ylabel("second component")
        plt.title(f"Formation Control - Simulation time = {tt}")
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
plt.figure("Animation")
animation(ZZ_at, NN, 500, r)

plt.show()