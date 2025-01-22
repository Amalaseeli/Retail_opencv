# Use the official Python image as the base image
FROM python:3.9-slim
# Set the working dir inside the container
WORKDIR /app
# Copy the files
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY best_model.pt /app/best_model.pt
COPY object_detection.py /app/object_detection.py

CMD ["python", "object_detection.py"]
