from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np


def generate_launch_description():
    NN = 6
    n_x = 2
    x_init = np.random.rand(n_x * NN)

    L = 2
    D = 2 * L
    distances = np.array(
        [
            [0, L, 0, D, 0, L],
            [L, 0, L, 0, D, 0],
            [0, L, 0, L, 0, D],
            [D, 0, L, 0, L, 0],
            [0, D, 0, L, 0, L],
            [L, 0, D, 0, L, 0],
        ]
    )

    Adj = distances > 0

    MAXITERS = 200
    COMM_TIME = 5e-2

    node_list = []

    for i in range(NN):
        N_ii = np.nonzero(Adj[:, i])[0].tolist()
        index_ii = i * n_x + np.arange(n_x)
        x_init_i = x_init[index_ii].tolist()
        distances_ii = distances[:, i].tolist()

        node_list.append(
            Node(
                package="formation_control",
                namespace=f"agent_{i}",
                executable="generic_agent",
                parameters=[
                    {
                        "id": i,
                        "communication_time": COMM_TIME,
                        "neighbors": N_ii,
                        "xzero": x_init_i,
                        "dist": distances_ii,
                        "maxT": MAXITERS,
                    }
                ],
                output="screen",
                prefix=f'xterm -title "agent_{i}" -hold -e',
            )
        )

    return LaunchDescription(node_list)
