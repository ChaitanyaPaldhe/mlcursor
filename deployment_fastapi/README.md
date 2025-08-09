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
   - Main endpoint: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## API Endpoints

- `GET /` - Health check and basic info
- `GET /health` - Detailed health check
- `POST /predict` - Make predictions

## Example Usage

### Using curl:
```bash
# Health check
curl http://localhost:8000/

# Make prediction (adjust features as needed)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

### Using Python requests:
```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [1.0, 2.0, 3.0, 4.0]}
)
print(response.json())
```

## Model Information
- **Model file:** outputs\models\random_forest_penguins.pkl
- **Port:** 8000
- **Generated:** 2025-08-09 18:51:41
