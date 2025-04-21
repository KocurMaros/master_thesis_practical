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
source /home/collab/virt_enviroments/facial_expressions/bin/activate
python3 /home/collab/collab_ws/src/facial_expression/scripts/camera_streamer_node.py

# This will run after the node exits or is interrupted
cleanup