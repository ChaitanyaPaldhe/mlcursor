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
