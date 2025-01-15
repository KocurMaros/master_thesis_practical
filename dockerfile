# Use CUDA 11.2 and CUDNN 8 runtime with Ubuntu 20.04 as the base image
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables for non-interactive installation and Python path
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9
ENV PATH /usr/local/cuda/bin:$PATH

# Install dependencies and set up Python 3.9 environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    python3.9 \
    python3.9-distutils \
    python3.9-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    libopenblas-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages with specified versions
# Upgrade pip and install required Python packages with specified versions
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install \
    dlib==19.24.2 \
    matplotlib==3.8.3 \
    numpy==1.26.4 \
    opencv_python==4.9.0.80 \
    pandas==2.2.2 \
    Pillow==10.3.0 \
    retina_face==0.0.14 \
    seaborn==0.13.2 \
    torch==2.1.2 \
    torchvision==0.16.2 \
    tqdm==4.66.1 \
    urllib3==2.2.1 \
    jupyter
# Downgrade protobuf to resolve MediaPipe and TensorFlow compatibility issues
RUN python3.9 -m pip install protobuf==3.20.*

# Set Python 3.9 as the default Python version and link pip
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set the default working directory inside the container
WORKDIR /workspace

# Copy local files to the container's workspace directory
COPY . .

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Command to run Jupyter Notebook when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
