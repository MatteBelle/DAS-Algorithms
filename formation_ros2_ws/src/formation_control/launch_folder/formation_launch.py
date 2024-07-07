from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx
import numpy as np

def phi(z):
    return z
# Step 2: Define the nonlinear transformation function phi
def sigma(z):
    return np.mean(phi(z), axis=0)
def cost_function(z, r, s, gamma, delta):
    return gamma * np.linalg.norm(z - r)**2 + delta * np.linalg.norm(z - s)**2
def grad_function_1(z, r, s, gamma, delta):
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1])

def grad_function_2(z, s, delta):
    return 2 * delta * (z - s)

def grad_phi(z):
    return 1

def generate_launch_description():

    np.random.seed(0)
    # Parameters
    NN = 10  # Number of points
    d = 2    # Dimension of the input space
    q = 4
    gamma = 0.5
    delta = 0.5
    # Step 1: Generate a dataset
    r = np.random.randn(NN, d)

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
    MAXITERS = 300
    dd = 2

    ZZ_at = np.random.randn(MAXITERS, NN, dd)
    SS_at = np.zeros((MAXITERS, NN, dd))
    VV_at = np.zeros((MAXITERS, NN, dd))
    for ii in range(NN):
        SS_at[0, ii] = phi(ZZ_at[0, ii])
        VV_at[0, ii] = grad_function_2(ZZ_at[0, ii], SS_at[0, ii], delta)


    cost_at = np.zeros((MAXITERS))
    gradients_norm = np.zeros((MAXITERS))
    alpha = 1e-2

    COMM_TIME = 5e-2

    node_list = []

    for ii in range(NN):
        #print(ii)
        N_ii = np.nonzero(Adj[ii])[0]
        ZZ_at_0_ii = np.array(ZZ_at[0, ii])
        SS_at_0_ii = np.array(SS_at[0, ii])
        VV_at_0_ii = np.array(VV_at[0, ii])
        r_ii = np.array(r[ii])
        AA_ii = float(AA[ii, ii])
        print(AA_ii)
        AA_neighbors = []
        for neighbor in N_ii:
            AA_neighbors.append(AA[ii, neighbor])
        node_list.append(
            Node(
                package="formation_control",
                namespace=f"agent_{ii}",
                executable="generic_agent",
                parameters=[
                    {
                        "id": ii,
                        "communication_time": COMM_TIME,
                        "neighbors": N_ii.tolist(),
                        "zz_init": ZZ_at_0_ii.tolist(),
                        "ss_init": SS_at_0_ii.tolist(),
                        "vv_init": VV_at_0_ii.tolist(),
                        "r": r_ii.tolist(),
                        "gamma": gamma,
                        "delta": delta,
                        "alpha": alpha,
                        "AA": AA_ii,
                        "AA_neighbors": AA_neighbors,
                        "maxT": MAXITERS,
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{ii}" -hold -e',
            )
        )

    return LaunchDescription(node_list)
