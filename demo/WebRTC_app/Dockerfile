# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing pyc files to disc
# and buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# libsndfile1 and ffmpeg are required for audio processing (librosa, soundfile)
# git and g++ are often needed for installing python packages
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "nemo_toolkit[all]"

# Download the model during build
ARG HF_TOKEN
ARG BASE_MODEL

RUN huggingface-cli login --token $HF_TOKEN
RUN huggingface-cli download $BASE_MODEL

# Set the model path environment variable
ENV MODEL_PATH=/app/model

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 9009 available to the world outside this container
EXPOSE 9009

# Run app.py when the container launches
CMD ["python", "app.py"]
