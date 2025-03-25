#!/bin/bash
# filepath: /home/laptop/school/dp/master_thesis_practical/ros_package/facial_expression/run_facial_expression.sh

# Function to clean up processes on exit
cleanup() {
    echo "Cleaning up processes..."
    # Kill any background processes created by this script
    jobs -p | xargs -r kill -9
    exit 0
}

# Set trap for Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM EXIT

# Activate virtual environment
source /home/laptop/school/dp/virt_env/bin/activate
python3 /home/laptop/school/dp/master_thesis_practical/ros_package/facial_expression/scripts/prediction.py &
# Source ROS2 workspace
source /opt/ros/humble/setup.bash
source /home/laptop/school/dp/master_thesis_practical/ros_package/facial_expression/install/setup.bash 

# Run the ROS2 node
ros2 run facial_expression prediction_node

# This will run after the node exits or is interrupted
cleanup