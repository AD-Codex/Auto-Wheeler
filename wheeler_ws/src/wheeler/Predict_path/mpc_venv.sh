#!/bin/bash

# Activate the Python virtual environment
source /home/jetson/ROS/ROBOT2/src/robocop_control/Predict_path/mpc_ws/bin/activate
exec python "$@"
