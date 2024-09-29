FROM amd64/python:3.7-buster

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install numpy==1.19.5 Cython==0.29.30 \
    && pip install pystan==2.19.1.1 \
    && pip install -r requirements.txt \
    && pip install prophet==1.0.1

# Copy the rest of the application
COPY . /app

EXPOSE 8000

ENV NAME sample

# Run gunicorn when the container launches and bind port 8000 from app.py
CMD ["gunicorn", "--worker-class", "gthread", "--workers", "3", "-b", ":8000", "app:app"]
