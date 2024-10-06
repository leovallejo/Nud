# Use an official lightweight Python runtime as a parent image
FROM python:3.9-slim-bullseye

# Set environment variables to prevent Python from writing .pyc files and to buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building Python packages and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies without caching to reduce image size
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the models directory and ensure correct permissions
RUN mkdir -p /app/models

# Create a non-root user for running the application
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# Expose port 8000 for the Flask application
EXPOSE 8000

# Define the default command to run the Flask app using Gunicorn with multiple workers and threads
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app", "--workers", "4", "--threads", "2"]
