# Launch file for the formation control algorithm in 2.1
from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx
import numpy as np

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

def sigma(z):
    '''
    Nonlinear transformation function sigma
    Input:
    z: input vector
    Output:
    mean: mean of phi(z)
    '''
    return np.mean(phi(z), axis=0)

def cost_function(z, r, s, gamma, delta):
    '''
    Cost function to minimize
    Input:
    z: input vector
    r: target vector
    s: baricenter vector
    gamma: gamma parameter (weight of the target)
    delta: delta parameter (weight of the baricenter)
    Output:
    cost: cost value
    '''
    return gamma * np.linalg.norm(z - r)**2 + delta * np.linalg.norm(z - s)**2

def grad_function_1(z, r, s, gamma, delta):
    '''
    Gradient of the cost function
    Input:
    z: input vector
    r: target vector
    s: baricenter vector
    gamma: gamma parameter (weight of the target)
    delta: delta parameter (weight of the baricenter)
    Output:
    grad_1: gradient of the cost function of the first component
    grad_2: gradient of the cost function of the second component
    '''
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1])

def grad_function_2(z, s, delta):
    '''
    Gradient of the cost function
    Input:
    z: input vector
    s: baricenter vector
    delta: delta parameter (weight of the baricenter)
    Output:
    grad: gradient of the cost function
    '''
    return 2 * delta * (z - s)

def grad_phi(z):
    '''
    Gradient of the nonlinear transformation function phi
    Input:
    z: input vector
    Output:
    1: gradient of the output vector
    '''
    return 1

# Launch description
def generate_launch_description():

    # Parameters
    np.random.seed(0)
    NN = 10  # Number of agents
    MAXITERS = 300
    dd = 2      # Dimension of the input space
    q = 4       # Dimension of the output space
    gamma = 0.5 # Weight of the targets
    delta = 0.5 # Weight of the baricenter
    r = np.random.randn(NN, dd) # Targets (random)

    # Graph definition
    #G = nx.cycle_graph(NN)
    #G = nx.path_graph(NN)
    G = nx.star_graph(NN-1)

    I_NN = np.eye(NN)
    Adj = nx.adjacency_matrix(G)
    Adj = Adj.toarray()
    AA = np.zeros(shape=(NN, NN))

    # Initialization of the matrix AA
    for ii in range(NN):
        N_ii = np.nonzero(Adj[ii])[0]
        deg_ii = len(N_ii)
        for jj in N_ii:
            deg_jj = len(np.nonzero(Adj[jj])[0])
            AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

    AA += I_NN - np.diag(np.sum(AA, axis=0))

    # Initialization of the variables: Metropolis-Hastings weights
    ZZ_at = np.random.randn(MAXITERS, NN, dd) # Agents
    SS_at = np.zeros((MAXITERS, NN, dd)) # Baricenter
    VV_at = np.zeros((MAXITERS, NN, dd)) # Gradient of the cost function
    for ii in range(NN):
        SS_at[0, ii] = phi(ZZ_at[0, ii])
        VV_at[0, ii] = grad_function_2(ZZ_at[0, ii], SS_at[0, ii], delta)

    alpha = 1e-2 # Step size
    COMM_TIME = 5e-2 # Communication time
    node_list = [] # List of nodes (agents)

    for ii in range(NN):
        # Initialization of the variables to be sent to the agents
        N_ii = np.nonzero(Adj[ii])[0]
        ZZ_at_0_ii = np.array(ZZ_at[0, ii])
        SS_at_0_ii = np.array(SS_at[0, ii])
        VV_at_0_ii = np.array(VV_at[0, ii])
        r_ii = np.array(r[ii])
        AA_ii = float(AA[ii, ii])
        AA_neighbors = []
        for neighbor in N_ii:
            AA_neighbors.append(AA[ii, neighbor])
        node_list.append(
            Node(
                package="formation_control",
                namespace=f"agent_{ii}",
                executable="2_1_agent",
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
                        "q": q,
                        "NN": NN,
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{ii}" -hold -e',
            )
        )

    return LaunchDescription(node_list)
