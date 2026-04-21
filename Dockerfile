# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Train the model during image build
RUN python app/model.py

# Expose port
EXPOSE 5000

# Start the Flask app
CMD ["python", "app/predict.py"]