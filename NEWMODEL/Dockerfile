# Use a slim base image for reduced size and improved security
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1  # Prevent .pyc files from being created
ENV PYTHONUNBUFFERED=1           # Ensure logs are printed in real-time

# Set the working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose port 8000 for the application
EXPOSE 8000

# Run the application using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
