# DAS-Algorithms
Implementation of distributed algorithms (Distributed and Autonomous System) for classification and more.

## Project Structure:
```
DAS-Algorithms
│   README.md
│
└───Task2.ipynb
│
└───Task2.3.py
│
└───Task2.1
│   │   Task2.1.py
│   │   Task2.1_circle_target.py
│   │   Task2.1_square_target.py
│   │   Task2.1_simply_4.py
|
└───formation_ros2_ws
│   │   launch_folder
|   |   |   formation_launch_2_1.launch
|   |   |   formation_launch_2_3.launch
│   │   formation_control
|   |   |   the_agent_2_1.py
|   |   |   the_agent_2_3.py

```
## How to run the code:
- Install the required packages:
```
pip install -r requirements.txt
```
- Run the Jupyter Notebook for Task 1:
```
jupyter notebook Task1.ipynb
```
- Run the Python script for Task 2.1:
```
python Task2.1/Task2.1.py
python Task2.1/Task2.1_circle_target.py
python Task2.1/Task2.1_square_target.py
python Task2.1/Task2.1_simply_4.py
```
- Run the Python script for Task 2.3:
```
python Task2.3.py
```

## How to run the ROS 2 package:
- Install ROS 2 Foxy and the GUI tools.
- Go into the ws folder and build the package:
```
cd DAS-Algorithms/formation_ros2_ws
colcon build --symlink-install --packages-select formation_control
```
- Source the ROS 2 workspace:
```
source /opt/ros/foxy/setup.bash
. install/setup.bash
```
- Run the launch file for Task 2.1:
```
ros2 launch formation_control formation_launch_2_1.launch.py
```
- Run the launch file for Task 2.3:
```
ros2 launch formation_control formation_launch_2_3.launch.py
```
### Note: To visualize the robots' movements in Rviz, run the following command in a new terminal:
```
source /opt/ros/foxy/setup.bash
ros2 run rviz2 rviz2
```
- Load the configuration file found in DAS-Algorithms (task2_1_rviz or task2_3_rviz) to visualize the robots' movements.

- Enjoy the simulation!

## Project Description:
This report presents the solutions to the course project assignments for Distributed Autonomous Systems 2023-24. The project comprises two main tasks: Distributed Classification via Logistic Regression and Aggregative Optimization for Multi-Robot Systems.

In the first task, we implement a distributed classification algorithm using Logistic Regression across multiple agents. The agents collaboratively determine a nonlinear classifier for a dataset divided into subsets. The Gradient Tracking algorithm is employed to achieve consensus optimization, and the performance is evaluated through simulations, assessing the convergence to a stationary point, cost function evolution, and classification accuracy.

The second task focuses on the control of multi-robot systems using the Aggregative Tracking algorithm. Each robot aims to maintain a tight formation while moving towards individual targets in a two-dimensional environment. We develop a Python script to simulate this behavior, tuning cost functions for various scenarios, and provide an animated visualization of the robots' movements. Additionally, a ROS 2 package is created to implement the algorithm in a real-world robotic system, including experiments to navigate corridors without collisions.

Both tasks demonstrate the effectiveness of distributed algorithms in classification and control, validated through extensive simulations and real-world implementations in ROS 2. The results highlight the robustness of the proposed solutions in achieving desired outcomes in distributed autonomous systems.

## Authors:
- [Matteo Belletti]
- [Alessandro Pasi]
- [Stricescu Razvan Ciprian]