from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='facial_expresion',
            executable='prediction_node',
            output='screen'
        ),
        Node(
            package='your_cpp_package',
            executable='prediction_subscriber',
            output='screen'
        ),
    ])
