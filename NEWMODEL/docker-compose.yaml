version: '3.8'

services:
  redis:
    image: "redis:6.2-alpine"
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  inference:
    container_name: inference-hf
    build:
      context: .
      dockerfile: Dockerfile
    command: gunicorn -b 0.0.0.0:8000 app:app --workers 4 --threads 2
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./worker-data:/app/worker-data
    depends_on:
      redis:
        condition: service_healthy
    environment:
      - FLASK_ENV=production
      - RATE_LIMIT=100 per hour
    restart: unless-stopped

  worker:
    container_name: celery-worker
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A app.celery worker --loglevel=info
    volumes:
      - ./models:/app/models
      - ./worker-data:/app/worker-data
    depends_on:
      redis:
        condition: service_healthy
    env_file:
      - ./worker-data/env_file
    environment:
      - CELERY_WORKER_CONCURRENCY=4
    restart: unless-stopped

volumes:
  redis-data:
  models:
  worker-data:
