# Use the official Python image as the base image
FROM python:3.9-slim
# Set the working dir inside the container
WORKDIR /app
# Copy the files
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

COPY best_model.pt /app/best_model.pt
COPY object_detection.py /app/object_detection.py

CMD ["python", "object_detection.py"]
