from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List, Union
import os

# Load the trained model
try:
    model = joblib.load("G:\Codes\mlcursor\outputs\models\logistic_regression_iris.pkl")
    print(f"✅ Model loaded successfully from: G:\Codes\mlcursor\outputs\models\logistic_regression_iris.pkl")
except Exception as e:
    print(f"❌ Error loading model from G:\Codes\mlcursor\outputs\models\logistic_regression_iris.pkl: {e}")
    raise

app = FastAPI(
    title="ML Model API",
    description="API for serving machine learning model predictions",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: List[Union[float, int]]
    
class PredictionResponse(BaseModel):
    prediction: List[Union[float, int]]
    model_info: str

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "FastAPI ML Model is running",
        "model_path": "G:\Codes\mlcursor\outputs\models\logistic_regression_iris.pkl",
        "status": "healthy",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    try:
        # Try to access the model
        model_type = type(model).__name__
        return {
            "status": "healthy",
            "model_type": model_type,
            "model_loaded": True
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "model_loaded": False
        }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make predictions using the loaded model"""
    try:
        # Validate input
        if not request.features:
            raise HTTPException(status_code=400, detail="Features list cannot be empty")
        
        # Convert to numpy array and reshape for single prediction
        features_array = np.array([request.features])
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(prediction, 'tolist'):
            prediction_list = prediction.tolist()
        else:
            prediction_list = list(prediction)
        
        return PredictionResponse(
            prediction=prediction_list,
            model_info=f"Model: {type(model).__name__}"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input features: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI ML Model Server...")
    print(f"📊 Model: G:\Codes\mlcursor\outputs\models\logistic_regression_iris.pkl")
    print(f"🌐 Server will run on: http://localhost:8080")
    print(f"📖 API Documentation: http://localhost:8080/docs")
    print(f"⚡ Interactive API: http://localhost:8080/redoc")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )
