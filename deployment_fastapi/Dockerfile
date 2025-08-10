FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and model files
COPY main.py .
COPY logistic_regression_iris.pkl .
COPY logistic_regression_iris_encoders.pkl .
COPY logistic_regression_iris_scaler.pkl .

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
