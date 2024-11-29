# Use Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variable for Docker
ENV RUNNING_IN_DOCKER=true

# Expose the application port
EXPOSE 60000

# Start the application
CMD ["python", "main.py"]
