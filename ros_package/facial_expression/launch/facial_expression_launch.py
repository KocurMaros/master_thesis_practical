import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

# Path to the virtual environment's Python binary
VENV_PYTHON = "/home/laptop/school/dp/virt_env/bin/python3"

def generate_launch_description():
    return LaunchDescription([
        # Run ROS2 node using Python from the virtual environment
        ExecuteProcess(
            cmd=[VENV_PYTHON, "-m", "ros2cli", "run", "facial_expression", "prediction_node"],
            shell=False
        ),
        # Start the ROS2 node
        Node(
            package='facial_expression',
            executable='prediction_node',
            name='facial_expression_node',
            output='screen'
        )
    ])
