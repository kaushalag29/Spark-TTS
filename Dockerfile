# Base image with Anaconda
FROM continuumio/miniconda3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCKER=true

# Set working directory
WORKDIR /app

# Install system dependencies
# git is needed for some pip installs from git repos
# ffmpeg is a common dependency for audio processing libraries
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && apt-get clean

# Copy the application files into the container
COPY . .

# Create a conda environment for the application
# Spark-TTS requires Python >= 3.9, using 3.11 for consistency
RUN conda create -n sparktts python=3.11 -y

# Activate the conda environment for subsequent commands
SHELL ["conda", "run", "-n", "sparktts", "/bin/bash", "-c"]

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install huggingface_hub for model downloading
RUN pip install huggingface_hub

# Install the Spark-TTS package in development mode
RUN pip install -e .

# Download Spark-TTS model from HuggingFace Hub if it doesn't exist
RUN mkdir -p pretrained_models && \
    if [ ! -d "pretrained_models/Spark-TTS-0.5B" ]; then \
        echo "Downloading Spark-TTS model from HuggingFace Hub..."; \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='pretrained_models/Spark-TTS-0.5B')"; \
    else \
        echo "Spark-TTS model directory already exists."; \
    fi

# Verify model directory exists and has content
RUN python -c "import os; model_dir='pretrained_models/Spark-TTS-0.5B'; print('Spark-TTS model check:'); print('Model directory exists:', os.path.exists(model_dir)); print('Directory contents:', os.listdir(model_dir) if os.path.exists(model_dir) else 'No model directory')"

# Expose the port the server will run on (port 8010 as specified in server.py)
EXPOSE 8010

# Command to run the application server with conda environment activated
CMD ["conda", "run", "-n", "sparktts", "python", "server.py"] 