from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat


def grad_function_1(z, r, s, gamma, delta):
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1])

def grad_function_2(z, s, delta):
    return 2 * delta * (z - s)

def grad_phi(z):
    return 1

def phi(z):
    return z

def gradient_tracking(zz_at, ss_at, vv_at, alpha, r, AA, AA_neighbors, vv_at_neighbors, ss_at_neighbors, delta):
    zz_next = zz_at - alpha * (grad_function_1(zz_at, r, ss_at) + grad_phi(zz_at) * vv_at)
    ss_next = AA * ss_at
    vv_next = AA * vv_at
    for jj in AA_neighbors:
        vv_next += AA * vv_at_neighbors[jj]
        ss_next += AA * ss_at_neighbors[jj]

    ss_next += phi(zz_next) - phi(zz_at)
    vv_next += grad_function_2(zz_next, ss_next, delta) - grad_function_2(zz_at, ss_at, delta)
    return zz_next, ss_next, vv_next

def formation_control_vector_field(
    DeltaT,
    XX_ii,
    N_ii,
    data,
    dist_ii,
):
    XX_iii_dot = np.zeros(XX_ii.shape)

    for jj in N_ii:
        XX_jj = np.array(data[jj].pop(0)[1:])

        dV_ij = (np.linalg.norm(XX_ii - XX_jj) ** 2 - dist_ii[jj] ** 2) * (
            XX_ii - XX_jj
        )
        XX_iii_dot -= dV_ij

    return XX_ii + DeltaT * XX_iii_dot


class Agent(Node):
    def __init__(self):
        super().__init__(
            "agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.agent_id = self.get_parameter("id").value
        self.neighbors = self.get_parameter("neighbors").value
        self.zz_init = self.get_parameter("zz_init").value
        self.ss_init = self.get_parameter("ss_init").value
        self.vv_init = self.get_parameter("vv_init").value
        self.r = self.get_parameter("r").value
        self.gamma = self.get_parameter("gamma").value
        self.delta = self.get_parameter("delta").value
        self.alpha = self.get_parameter("alpha").value
        self.AA = self.get_parameter("AA").value
        self.AA_neighbors = self.get_parameter("AA_neighbors").value
        self.maxIters = self.get_parameter("maxT").value

        self.get_logger().info(f"I am agent: {self.agent_id}")

        communication_time = self.get_parameter("communication_time").value
        self.DeltaT = communication_time / 10

        self.t = 0

        for j in self.neighbors:
            print(self.neighbors)
            self.create_subscription(
                MsgFloat, f"/topic_{j}", self.listener_callback, 10
            )

        self.publisher = self.create_publisher(
            MsgFloat, f"/topic_{self.agent_id}", 10
        )  # topic_i

        self.timer = self.create_timer(communication_time, self.timer_callback)

        self.received_data = {j: [] for j in self.neighbors}
        self.vv_at_neighbors = {j: np.zeros_like(self.vv_init) for j in self.neighbors}
        self.ss_at_neighbors = {j: np.zeros_like(self.ss_init) for j in self.neighbors}

    def listener_callback(self, msg):
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.received_data[j].append(msg_j)
        self.vv_at_neighbors[j] = np.array(msg_j[2])
        self.ss_at_neighbors[j] = np.array(msg_j[1])
        return None

    def timer_callback(self):
        msg = MsgFloat()

        if self.t == 0:
            msg.data = [float(self.agent_id), float(self.t)] + self.ss_init.tolist() + self.vv_init.tolist()
            self.publisher.publish(msg)

            ss_to_string = f"{np.array2string(self.ss_init, precision=4, floatmode='fixed', separator=', ')}"
            vv_to_string = f"{np.array2string(self.vv_init, precision=4, floatmode='fixed', separator=', ')}"

            self.get_logger().info(f"Iter: {self.t} x_{self.agent_id}: {ss_to_string, vv_to_string}")

            self.t += 1
        else:
            # all_received = all(
            #     self.t - 1 == self.received_data[j][0][0] for j in self.neighbors
            # )
            all_received = False
            if all(len(self.received_data[j]) > 0 for j in self.neighbors):
                all_received = all(
                    self.t - 1 == self.received_data[j][0][0] for j in self.neighbors
                )

            if all_received:
                self.zz_init, self.ss_init, self.vv_init = gradient_tracking(
                    self.zz_init,
                    self.ss_init,
                    self.vv_init,
                    self.alpha,
                    self.r,
                    self.AA,
                    self.AA_neighbors,
                    self.vv_at_neighbors,
                    self.ss_at_neighbors,
                    self.delta,
                )

                msg.data = [float(self.agent_id), float(self.t)] + self.x_i.tolist()
                self.publisher.publish(msg)

                x_i_string = f"{np.array2string(self.x_i, precision=4, floatmode='fixed', separator=', ')}"
                self.get_logger().info(
                    f"Iter: {self.t} x_{self.agent_id}: {x_i_string}"
                )

                self.t += 1

                if self.t > self.maxIters:
                    print("\nMax iters reached")
                    sleep(3)
                    self.destroy_node()


def main():
    print("Starting agent")
    rclpy.init()
    agent = Agent()
    sleep(1)
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
