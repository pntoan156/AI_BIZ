# Use Python 3.13 slim base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install pdm

# Copy PDM files
COPY pyproject.toml pdm.lock ./

# Copy application code
COPY . .

# Install dependencies using PDM
RUN pdm install --prod

# Expose the port the app runs on
EXPOSE 8500

# Command to run the application
CMD ["pdm", "run", "start"]