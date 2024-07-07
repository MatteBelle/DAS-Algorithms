from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from geometry_msgs.msg import PoseStamped, PointStamped
from builtin_interfaces.msg import Time


def grad_function_1(z, r, s, gamma, delta):
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1])

def grad_function_2(z, s, delta):
    return 2 * delta * (z - s)

def grad_phi(z):
    return 1

def phi(z):
    return z

def gradient_tracking(zz_at, neighbors, ss_at, vv_at, alpha, r, AA, AA_neighbors, vv_at_neighbors, ss_at_neighbors, delta, gamma):
    zz_next = zz_at - alpha * (grad_function_1(zz_at, r, ss_at, gamma, delta) + grad_phi(zz_at) * vv_at)
    ss_next = AA * ss_at
    vv_next = AA * vv_at
    for jj, neighbor in zip(AA_neighbors, neighbors):
        vv_next += vv_at_neighbors[neighbor] * jj
        ss_next += ss_at_neighbors[neighbor] * jj

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
        print(f"Agent id: {self.agent_id}")
        self.neighbors = np.array(self.get_parameter("neighbors").value)
        print(f"Neighbors: {self.neighbors}")
        self.zz_init = np.array(self.get_parameter("zz_init").value)
        print(f"zz_init: {self.zz_init}")
        self.ss_init = np.array(self.get_parameter("ss_init").value)
        print(f"ss_init: {self.ss_init}")
        self.vv_init = np.array(self.get_parameter("vv_init").value)
        print(f"vv_init: {self.vv_init}")
        self.r = np.array(self.get_parameter("r").value)
        print(f"r: {self.r}")
        self.gamma = self.get_parameter("gamma").value
        print(f"gamma: {self.gamma}")
        self.delta = self.get_parameter("delta").value
        print(f"delta: {self.delta}")
        self.alpha = self.get_parameter("alpha").value
        print(f"alpha: {self.alpha}")
        self.AA = self.get_parameter("AA").value
        print(f"AA: {self.AA}")
        self.AA_neighbors = np.array(self.get_parameter("AA_neighbors").value)
        print(f"AA_neighbors: {self.AA_neighbors}")
        self.maxIters = self.get_parameter("maxT").value
        print(f"maxIters: {self.maxIters}")

        self.get_logger().info(f"I am agent: {self.agent_id}")

        communication_time = self.get_parameter("communication_time").value
        print(f"Communication time: {communication_time}")
        self.DeltaT = communication_time / 10

        self.t = 0

        for j in self.neighbors:
            print(self.neighbors)
            self.create_subscription(
                MsgFloat, f"/topic_{j}", self.listener_callback, 10
            )

        for i in range(10):
            self.create_subscription(
                MsgFloat, f"/topic_{i}", self.plotter_callback, 10
            )

        self.publisher = self.create_publisher(
            MsgFloat, f"/topic_{self.agent_id}", 10
        )  # topic_i

        self.timer = self.create_timer(communication_time, self.timer_callback)

        self.received_data = {j: [] for j in self.neighbors}
        self.plot_data = {j: [] for j in range(10)}
        self.vv_at_neighbors = {j: np.zeros_like(self.vv_init) for j in self.neighbors}
        self.ss_at_neighbors = {j: np.zeros_like(self.ss_init) for j in self.neighbors}

        # RViz publisher
        self.pose_pub = self.create_publisher(PointStamped, f"/pose_{self.agent_id}", 10)
        self.timer_rviz = self.create_timer(1, self.publish_pose)
        self.target_pub = self.create_publisher(PointStamped, f"/target_{self.agent_id}", 10)
        self.timer_target = self.create_timer(1, self.publish_target)


    def publish_pose(self):
        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.point.x = self.zz_init[0]
        msg.point.y = self.zz_init[1]
        self.pose_pub.publish(msg)

    def publish_target(self):
        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.point.x = self.r[0]
        msg.point.y = self.r[1]
        # change color of target to green
        self.target_pub.publish(msg)

    def listener_callback(self, msg):
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.received_data[j].append(msg_j)
        self.vv_at_neighbors[j] = np.array(msg_j[1:3])
        self.ss_at_neighbors[j] = np.array(msg_j[3:5])
        return None
    
    def plotter_callback(self, msg):
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])
        self.plot_data[j].append(msg_j)
        return None

    def timer_callback(self):
        msg = MsgFloat()

        if self.t == 0:
            msg.data = [float(self.agent_id), float(self.t)] + self.ss_init.tolist() + self.vv_init.tolist()
            print(f"Data sent: {msg.data}")
            print("++++++++++++")
            print([float(self.agent_id), float(self.t)])
            print(self.ss_init.tolist())
            print(self.vv_init.tolist())
            print("++++++++++++")
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
            #last_t_neighbors = [self.received_data[j][-1][0] for j in self.neighbors]
            if all(len(self.received_data[j]) > 0 for j in self.neighbors):
                all_received = all(
                    self.t - 1 == int(self.received_data[j][-1][0]) for j in self.neighbors
                )

            if all_received:
                self.zz_init, self.ss_init, self.vv_init = gradient_tracking(
                    self.zz_init,
                    self.neighbors,
                    self.ss_init,
                    self.vv_init,
                    self.alpha,
                    self.r,
                    self.AA,
                    self.AA_neighbors,
                    self.vv_at_neighbors,
                    self.ss_at_neighbors,
                    self.delta,
                    self.gamma,
                )
                sleep(0.5)
                msg.data = [float(self.agent_id), float(self.t)] + self.ss_init.tolist() + self.vv_init.tolist()
                self.publisher.publish(msg)

                ss_to_string = f"{np.array2string(self.ss_init, precision=4, floatmode='fixed', separator=', ')}"
                vv_to_string = f"{np.array2string(self.vv_init, precision=4, floatmode='fixed', separator=', ')}"

                self.get_logger().info(f"Iter: {self.t} x_{self.agent_id}: {ss_to_string, vv_to_string}")

                self.t += 1

                if self.t > self.maxIters:
                    print("\nMax iters reached")
                    sleep(3)
                    # save plot data to file
                    if self.agent_id == 0:
                        f = open(f"plot_data.txt", "w")
                        f.write(str(self.plot_data))
                        f.close()
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
