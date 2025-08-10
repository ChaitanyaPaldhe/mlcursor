# FastAPI ML Model Deployment with Preprocessing

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
   - Preprocessing info: http://localhost:8000/preprocessing-info


## Automatic Preprocessing

This API automatically applies preprocessing to your input features:

1. **Encoding**: Categorical features are encoded using the trained encoder
2. **Scaling**: Numeric features are scaled using the trained scaler

**Important**: Send raw, unprocessed features to the API. The preprocessing will be applied automatically.

### Preprocessing Files:
- Encoder: `logistic_regression_iris_encoders.pkl`
- Scaler: `logistic_regression_iris_scaler.pkl`


## Model Information
- **Model file:** logistic_regression_iris.pkl
- **Encoder file:** logistic_regression_iris_encoders.pkl
- **Scaler file:** logistic_regression_iris_scaler.pkl
- **Port:** 8000
- **Auto-preprocessing:** ✅ Enabled

## Example Usage

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"batch_features": [[5.1, 3.5, 1.4, 0.2], [6.0, 3.0, 4.0, 1.2]]}'
```
