FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY main.py .
COPY random_forest_penguins.pkl .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
