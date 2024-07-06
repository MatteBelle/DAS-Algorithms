from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat


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
        self.x_i = np.array(self.get_parameter("xzero").value)
        self.dist_ii = np.array(self.get_parameter("dist").value)
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

    def listener_callback(self, msg):
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.received_data[j].append(msg_j)
        return None

    def timer_callback(self):
        msg = MsgFloat()

        if self.t == 0:
            msg.data = [float(self.agent_id), float(self.t)] + self.x_i.tolist()
            self.publisher.publish(msg)

            x_i_string = f"{np.array2string(self.x_i, precision=4, floatmode='fixed', separator=', ')}"

            self.get_logger().info(f"Iter: {self.t} x_{self.agent_id}: {x_i_string}")

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
                self.x_i = formation_control_vector_field(
                    self.DeltaT,
                    self.x_i,
                    self.neighbors,
                    self.received_data,
                    self.dist_ii,
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
    rclpy.init()

    agent = Agent()
    sleep(1)
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
