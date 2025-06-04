# Emotion Recognition System

This project provides a pipeline for training and deploying a facial emotion recognition model using Docker and ROS2. It includes a web interface for model training and real-time emotion recognition with ROS2 integration.

---

## Features

- Web-based interface for training emotion recognition models
- Dockerized environment for easy setup and reproducibility
- ROS2 integration for real-time emotion detection from camera streams
- Visualization of results via a web dashboard and ROS2 topics

---

## Requirements

### For Model Training

- Docker
- Docker Compose
- At least 8 GB RAM and 4 GB GPU memory
- Minimum 10 GB free disk space

### For ROS2 Deployment

- Ubuntu 22.04 LTS
- ROS2 Humble
- ROS2-compatible camera
- Pre-trained model (output from the training process)

### For XIMEA camera

 - Install XIMEA package from url:
 ```
 https://www.ximea.com/support/wiki/apis/ximea_linux_software_package
```
---


## Notes for XIMEA Camera Users

If you are using a XIMEA camera:

- You must copy the `xiAPI` library to your Python virtual environment.
- At every boot, you must delete the default buffer size for the camera on Linux by running:

```
$ sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb >/dev/null <<<0
```

## Getting Started

### 1. Clone the Repository

```
$ git clone https://github.com/KocurMaros/master_thesis_practical
$ cd master_thesis_practical
```

---

### 2. Train the Model (via Docker)

#### Build Docker Container

```
$ docker-compose build
```

#### Launch the Container

```
$ docker-compose up
```

#### Access the Web Interface

Open your browser and go to:

```
$ http://localhost:5000
```

#### Start Training
- download dataset you want to use
- Select your jupyter notebook based on dataset you want train on.
- Set hyperparameters (epochs, batch size, learning rate)
- Click "Spustiť trénovanie" (Start Training)
- Monitor training progress and metrics

#### Export the Trained Model

After training, download the model by clicking "Exportovať model" (Export Model).

---

### 3. Deploy with ROS2

#### Prepare ROS2 Workspace

```
$ mkdir -p ~/ros2_ws/src
$ cd ~/ros2_ws/
$ cp -r /path/to/your/master_thesis_practical/facial_expression ~/ros2_ws/src/
$ python3.10 -m venv .virt_env
$ source .virt_env/bin/activate
$ pip install -r req.txt
$ cp /path/to/downloaded/ximea/package/api/Python/v3/ximea/ .virt_env/lib/python3.10/site-packages/ximea
```
#### Build the ROS2 Package

```
$ colcon build --symlink-install --packages-select emotion_recognition
```
#### Source the ROS2 Environment

```
$ source install/setup.bash
```
---

### 4. Run the System

#### Start Video Stream

```
$ ./scripts/video_stream.sh
```
#### Start Emotion Recognition

```
$ ./scripts/run_facial_expression.sh
```

This will launch:

- Face detection node
- Emotion prediction node
- Web server for result visualization

---

### 5. Access Results

- Web interface:  
  `http://localhost:8080`
- ROS2 topics:
  - Emotion predictions:  
    `ros2 topic echo /emotion_prediction`
  - Video stream detections:  
    `ros2 topic echo /rgb_image/ximea` or `ros2 topic echo /rgb_image/default` 

---

## Troubleshooting

- **ROS2 node issues:**  
  Check active nodes with `ros2 node list`
- **Prediction errors:**  
  Verify the model path in the configuration file

---

## Stopping the System

To stop all running nodes, press `Ctrl+C` in the terminal(s) running the scripts, or run:

---

## License

This project is part of a master's thesis. See the repository for licensing details.

---

## Author

Maros Kocur

---

*For more details, refer to the full user manual in the documentation or contact owner*