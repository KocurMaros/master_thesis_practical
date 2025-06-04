#!/bin/bash
# filepath: /home/collab/collab_ws/src/facial_expression/run_facial_expression.sh

# Function to clean up processes on exit
cleanup() {
    echo "Cleaning up processes..."
    # Kill any background processes created by this script
    jobs -p | xargs -r kill -9
    exit 0
}

# Set trap for Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM EXIT
# sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0
# Activate virtual environment
source /~/ros2_ws/.virt_env/facial_expressions/bin/activate
python3 ~/ros2_ws/src/facial_expression/scripts/prediction.py &
# Source ROS2 workspace
# source /opt/ros/humble/setup.bash

# Run the ROS2 node
ros2 run facial_expression prediction_node

# This will run after the node exits or is interrupted
cleanup