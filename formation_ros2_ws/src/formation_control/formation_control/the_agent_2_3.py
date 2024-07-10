from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker

def grad_function_1(z, r, s, w1, w2, epsilon, gamma, delta):
    c = 0
    if (z[0] < w1[1][0]):
        c = 2 * epsilon * (z[1] - (w1[0][1] + w2[0][1]) / 2)
    return 2 * gamma * (z[0] - r[0]) + 2 * delta * (z[0] - s[0]), 2 * gamma * (z[1] - r[1]) + 2 * delta * (z[1] - s[1]) + c

def grad_function_2(z, s, delta):
    return 2 * delta * (z - s)

def cost_function(z, r, s, w1, w2, gamma, delta, epsilon):
    return gamma * np.linalg.norm(z - r)**2 + delta * np.linalg.norm(z - s)**2 + epsilon * (np.linalg.norm(z[1] - (w1[0][1] + w2[0][1]) / 2)**2)

def grad_phi(z):
    return 1

def phi(z):
    return z

def gradient_tracking(zz_at, neighbors, ss_at, vv_at, alpha, r, AA, AA_neighbors, vv_at_neighbors, ss_at_neighbors, delta, gamma, epsilon, wall1, wall2, q): 
    gradients_k = np.zeros((q))

    zz_next = zz_at - alpha * (grad_function_1(zz_at, r, ss_at, wall1, wall2, epsilon, gamma, delta) + grad_phi(zz_at) * vv_at)
    ss_next = AA * ss_at
    vv_next = AA * vv_at
    for jj, neighbor in zip(AA_neighbors, neighbors):
        vv_next += vv_at_neighbors[neighbor] * jj
        ss_next += ss_at_neighbors[neighbor] * jj

    ss_next += phi(zz_next) - phi(zz_at)
     
    new_grad_3 = grad_function_2(zz_next, ss_next, delta)
    old_grad_3 = grad_function_2(zz_at, ss_at, delta)
    vv_next += new_grad_3 - old_grad_3

    old_grad_1, old_grad_2 = grad_function_1(zz_at, r, ss_at, wall1, wall2, epsilon, gamma, delta)
    new_grad_1, new_grad_2 = grad_function_1(zz_next, r, ss_next, wall1, wall2, epsilon, gamma, delta)
    
    cost_at = cost_function(zz_at, r, ss_at, wall1, wall2, gamma, delta, epsilon)
    gradients_k[0] += new_grad_1 - old_grad_1
    gradients_k[1] += new_grad_2 - old_grad_2
    gradients_k[2] += new_grad_3[0] - old_grad_3[0]
    gradients_k[3] += new_grad_3[1] - old_grad_3[1]
    
    return zz_next, ss_next, vv_next, cost_at, gradients_k

def plot_data_fn(data_to_plot, maxIters, q):
    # Get data for each agent (agent numbers are dictionary keys)
    cost_at_kk = np.zeros((maxIters))
    gradients_norm_kk = np.zeros((maxIters))

    for kk in range(maxIters):
        gradient_k = np.zeros((q))
        for agent in data_to_plot.keys():
            data = data_to_plot.get(agent)
            for ii, jj in zip(range(q), range(6, 6 + q)):
                gradient_k[ii] += data[kk][jj]
            cost_at_kk[kk] += data[kk][5]
        gradients_norm_kk[kk] = np.linalg.norm(gradient_k)
            
    # Plot data
    fig, ax = plt.subplots()
    ax.semilogy(np.arange(maxIters - 1), cost_at_kk[:-1])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost function")
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(maxIters - 1), gradients_norm_kk[:-1])
    plt.xlabel("Iterations")
    plt.ylabel("Gradients norm")
    plt.title("Gradients norm")
    ax.grid()
    plt.show()

def create_color_map(NN):
    colors = []
    for i in range(NN):
        colors.append((np.random.rand(), np.random.rand(), np.random.rand()))
    return colors

class Agent(Node):
    def __init__(self):
        super().__init__(
            "agent",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.agent_id = self.get_parameter("id").value
        self.neighbors = np.array(self.get_parameter("neighbors").value)
        self.zz_init = np.array(self.get_parameter("zz_init").value)
        self.ss_init = np.array(self.get_parameter("ss_init").value)
        self.vv_init = np.array(self.get_parameter("vv_init").value)
        self.r = np.array(self.get_parameter("r").value)
        self.gamma = self.get_parameter("gamma").value
        self.delta = self.get_parameter("delta").value
        self.alpha = self.get_parameter("alpha").value
        self.AA = self.get_parameter("AA").value
        self.AA_neighbors = np.array(self.get_parameter("AA_neighbors").value)
        self.maxIters = self.get_parameter("maxT").value


        self.wall1_1 = np.array(self.get_parameter("wall1_point1").value)
        self.wall1_2 = np.array(self.get_parameter("wall1_point2").value)
        self.wall1 = [self.wall1_1, self.wall1_2]
        self.wall2_1 = np.array(self.get_parameter("wall2_point1").value)
        self.wall2_2 = np.array(self.get_parameter("wall2_point2").value)
        self.wall2 = [self.wall2_1, self.wall2_2]
        self.epsilon = self.get_parameter("epsilon").value
        self.q = self.get_parameter("q").value
        self.nn = self.get_parameter("NN").value
        self.cost_at = 0
        self.gradients_norm = np.zeros((self.q))
        self.get_logger().info(f"I am agent: {self.agent_id}")

        communication_time = self.get_parameter("communication_time").value
        print(f"Communication time: {communication_time}")
        self.DeltaT = communication_time / 10

        self.t = 0

        for j in self.neighbors:
            print(self.neighbors)
            self.create_subscription(
                MsgFloat, f"/topic_{j}", self.listener_callback, 100
            )

        self.publisher = self.create_publisher(
            MsgFloat, f"/topic_{self.agent_id}", 100
        )  # topic_i

        self.timer = self.create_timer(communication_time, self.timer_callback)

        self.received_data = {j: [] for j in self.neighbors}
        if self.agent_id == 0:
            for j in range(self.nn):
                self.create_subscription(
                    MsgFloat, f"/topic_{j}", self.plotter_callback, 1000
            )
            self.plot_data = {j: [] for j in range(self.nn)}
        self.vv_at_neighbors = {j: np.zeros_like(self.vv_init) for j in self.neighbors}
        self.ss_at_neighbors = {j: np.zeros_like(self.ss_init) for j in self.neighbors}

        # RViz publisher
        self.pose_pub = self.create_publisher(Marker, f"/pose_{self.agent_id}", 10)
        self.timer_rviz = self.create_timer(1, self.publish_pose)
        self.target_pub = self.create_publisher(Marker, f"/target_{self.agent_id}", 10)
        self.timer_target = self.create_timer(1, self.publish_target)

        # RViz publisher for walls
        self.wall1_pub = self.create_publisher(Marker, f"/wall1_{self.agent_id}", 10)
        self.wall2_pub = self.create_publisher(Marker, f"/wall2_{self.agent_id}", 10)
        self.timer_wall1 = self.create_timer(1, self.publish_wall1)
        self.timer_wall2 = self.create_timer(1, self.publish_wall2)

        self.color = create_color_map(self.nn)[self.agent_id]


    def publish_pose(self):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.type = msg.SPHERE
        msg.action = msg.ADD
        msg.scale.x = 0.2
        msg.scale.y = 0.2
        msg.scale.z = 0.2
        msg.color.r = self.color[0]
        msg.color.g = self.color[1]
        msg.color.b = self.color[2]
        msg.color.a = 1.0
        msg.pose.position.x = self.zz_init[0]
        msg.pose.position.y = self.zz_init[1]
        self.pose_pub.publish(msg)

    def publish_target(self):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.type = msg.CUBE
        msg.action = msg.ADD
        msg.scale.x = 0.2
        msg.scale.y = 0.2
        msg.scale.z = 0.2
        msg.color.r = self.color[0]
        msg.color.g = self.color[1]
        msg.color.b = self.color[2]
        msg.color.a = 1.0
        msg.pose.position.x = self.r[0]
        msg.pose.position.y = self.r[1]
        self.target_pub.publish(msg)

    def publish_wall1(self):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.type = msg.LINE_STRIP
        msg.action = msg.ADD
        msg.scale.x = 0.1
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.points = []
        for point in self.wall1:
            msg.points.append(Point(x=point[0], y=point[1]))
        self.wall1_pub.publish(msg)
        

    def publish_wall2(self):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.type = msg.LINE_STRIP
        msg.action = msg.ADD
        msg.scale.x = 0.1
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.points = []
        for point in self.wall2:
            msg.points.append(Point(x=point[0], y=point[1]))
        self.wall2_pub.publish(msg)

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
            msg.data = [float(self.agent_id), float(self.t)] + self.ss_init.tolist() + self.vv_init.tolist() + [float(self.cost_at)] + self.gradients_norm.tolist()
            print(f"Data sent: {msg.data}")
            print("++++++++++++")
            print([float(self.agent_id), float(self.t)])
            print(self.ss_init.tolist())
            print(self.vv_init.tolist())
            print("++++++++++++")
            self.publisher.publish(msg)
            ss_to_string = f"{np.array2string(self.ss_init, precision=4, floatmode='fixed', separator=', ')}"
            vv_to_string = f"{np.array2string(self.vv_init, precision=4, floatmode='fixed', separator=', ')}"
            cost_to_string = f"{self.cost_at}"
            gradients_to_string = f"{np.array2string(self.gradients_norm, precision=4, floatmode='fixed', separator=', ')}"

            self.get_logger().info(f"Iter: {self.t} x_{self.agent_id}: {ss_to_string, vv_to_string, cost_to_string, gradients_to_string}")

            self.t += 1
        else:
            # all_received = all(
            #     self.t - 1 == self.received_data[j][0][0] for j in self.neighbors
            # )
            all_received = False
            if all(len(self.received_data[j]) > 0 for j in self.neighbors):
                all_received = all(
                    self.t - 1 == int(self.received_data[j][-1][0]) for j in self.neighbors
                )

            if all_received:
                self.zz_init, self.ss_init, self.vv_init, self.cost_at, self.gradients_norm = gradient_tracking(
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
                    self.epsilon,
                    self.wall1,
                    self.wall2,
                    self.q,
                )
                sleep(1)
                msg.data = [float(self.agent_id), float(self.t)] + self.ss_init.tolist() + self.vv_init.tolist() + [float(self.cost_at)] + self.gradients_norm.tolist()
                self.publisher.publish(msg)

                ss_to_string = f"{np.array2string(self.ss_init, precision=4, floatmode='fixed', separator=', ')}"
                vv_to_string = f"{np.array2string(self.vv_init, precision=4, floatmode='fixed', separator=', ')}"
                cost_to_string = f"{self.cost_at}"
                gradients_to_string = f"{np.array2string(self.gradients_norm, precision=4, floatmode='fixed', separator=', ')}"

                self.get_logger().info(f"Iter: {self.t} x_{self.agent_id}: {ss_to_string, vv_to_string, cost_to_string, gradients_to_string}")

                self.t += 1

                if self.t > self.maxIters:
                    print("\nMax iters reached")
                    sleep(5)
                    if self.agent_id == 0:
                        sleep(3)
                        plot_data_fn(self.plot_data, self.maxIters, self.q)
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
