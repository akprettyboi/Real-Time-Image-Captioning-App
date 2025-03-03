#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
fi

# Check if NVIDIA GPU is present
if nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing NVIDIA Docker..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
fi

# Pull the Docker image
echo "Pulling the application image..."
docker pull yourdockerhubusername/image-captioning-app:latest

# Create a convenient run script
echo '#!/bin/bash

if nvidia-smi &> /dev/null; then
    # GPU version
    docker run --gpus all \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=$DISPLAY \
        -v $HOME/.Xauthority:/root/.Xauthority \
        --device=/dev/video0:/dev/video0 \
        yourdockerhubusername/image-captioning-app:latest
else
    # CPU version
    docker run \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY=$DISPLAY \
        -v $HOME/.Xauthority:/root/.Xauthority \
        --device=/dev/video0:/dev/video0 \
        yourdockerhubusername/image-captioning-app:latest
fi' > run_app.sh

chmod +x run_app.sh

echo "Installation complete! Run './run_app.sh' to start the application." 