# FastAPI ML Model Deployment

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python main.py
   ```

3. **Access the API:**
   - Main endpoint: http://localhost:8080
   - API Documentation: http://localhost:8080/docs
   - Alternative docs: http://localhost:8080/redoc

## API Endpoints

- `GET /` - Health check and basic info
- `GET /health` - Detailed health check
- `POST /predict` - Make predictions

## Example Usage

### Using curl:
```bash
# Health check
curl http://localhost:8080/

# Make prediction (adjust features as needed)
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

### Using Python requests:
```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8080/predict",
    json={"features": [1.0, 2.0, 3.0, 4.0]}
)
print(response.json())
```

## Model Information
- **Model file:** G:\Codes\mlcursor\outputs\models\logistic_regression_iris.pkl
- **Port:** 8080
- **Generated:** 2025-08-10 11:29:07
