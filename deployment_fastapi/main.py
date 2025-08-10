from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_filename = "logistic_regression_iris.pkl"
model_path = os.path.join(SCRIPT_DIR, model_filename)

try:
    model = joblib.load(model_path)
    logger.info(f"✅ Model loaded successfully from: {model_path}")
except Exception as e:
    logger.error(f"❌ Error loading model from {model_path}: {e}")
    raise

# Load encoder
encoder = None
encoder_filename = "logistic_regression_iris_encoders.pkl"
encoder_path = os.path.join(SCRIPT_DIR, encoder_filename)

try:
    encoder = joblib.load(encoder_path)
    logger.info(f"✅ Encoder loaded successfully from: {encoder_path}")
except Exception as e:
    logger.warning(f"⚠️ Could not load encoder from {encoder_path}: {e}")
    encoder = None

# Load scaler
scaler = None
scaler_filename = "logistic_regression_iris_scaler.pkl"
scaler_path = os.path.join(SCRIPT_DIR, scaler_filename)

try:
    scaler = joblib.load(scaler_path)
    logger.info(f"✅ Scaler loaded successfully from: {scaler_path}")
except Exception as e:
    logger.warning(f"⚠️ Could not load scaler from {scaler_path}: {e}")
    scaler = None

app = FastAPI(
    title="ML Model API with Preprocessing",
    description="API for serving machine learning model predictions with automatic preprocessing",
    version="2.0.0"
)

class PredictionRequest(BaseModel):
    features: List[Union[float, int, str]]  # Allow mixed types for categorical features
    
class PredictionResponse(BaseModel):
    prediction: List[Union[float, int]]
    model_info: str
    preprocessing_applied: Dict[str, bool]
    input_shape: List[int]

class BatchPredictionRequest(BaseModel):
    batch_features: List[List[Union[float, int, str]]]

class BatchPredictionResponse(BaseModel):
    predictions: List[List[Union[float, int]]]
    model_info: str
    preprocessing_applied: Dict[str, bool]
    batch_size: int

def preprocess_features(features: List[Union[float, int, str]]) -> np.ndarray:
    """Apply preprocessing pipeline: encode -> scale"""
    try:
        # Convert to numpy array
        features_array = np.array([features])
        
        # Step 1: Apply encoding if encoder is available
        if encoder is not None:
            try:
                # Handle different encoder types
                if hasattr(encoder, 'transform'):
                    features_array = encoder.transform(features_array)
                elif hasattr(encoder, 'encode'):
                    features_array = encoder.encode(features_array)
                logger.info("✅ Encoding applied")
            except Exception as e:
                logger.warning(f"⚠️ Encoding failed: {e}")
        
        # Step 2: Apply scaling if scaler is available
        if scaler is not None:
            try:
                features_array = scaler.transform(features_array)
                logger.info("✅ Scaling applied")
            except Exception as e:
                logger.warning(f"⚠️ Scaling failed: {e}")
        
        return features_array
        
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}")

def preprocess_batch_features(batch_features: List[List[Union[float, int, str]]]) -> np.ndarray:
    """Apply preprocessing pipeline to batch of features"""
    try:
        # Convert to numpy array
        features_array = np.array(batch_features)
        
        # Step 1: Apply encoding if encoder is available
        if encoder is not None:
            try:
                if hasattr(encoder, 'transform'):
                    features_array = encoder.transform(features_array)
                elif hasattr(encoder, 'encode'):
                    features_array = encoder.encode(features_array)
                logger.info("✅ Batch encoding applied")
            except Exception as e:
                logger.warning(f"⚠️ Batch encoding failed: {e}")
        
        # Step 2: Apply scaling if scaler is available
        if scaler is not None:
            try:
                features_array = scaler.transform(features_array)
                logger.info("✅ Batch scaling applied")
            except Exception as e:
                logger.warning(f"⚠️ Batch scaling failed: {e}")
        
        return features_array
        
    except Exception as e:
        raise ValueError(f"Batch preprocessing failed: {str(e)}")

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "FastAPI ML Model with Preprocessing is running",
        "model_file": model_filename,
        "encoder_available": encoder is not None,
        "scaler_available": scaler is not None,
        "encoder_file": encoder_filename if encoder is not None else None,
        "scaler_file": scaler_filename if scaler is not None else None,
        "status": "healthy",
        "endpoints": ["/predict", "/predict/batch", "/health", "/docs", "/preprocessing-info"]
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    try:
        model_type = type(model).__name__
        encoder_type = type(encoder).__name__ if encoder else None
        scaler_type = type(scaler).__name__ if scaler else None
        
        return {
            "status": "healthy",
            "model_type": model_type,
            "encoder_type": encoder_type,
            "scaler_type": scaler_type,
            "model_loaded": True,
            "encoder_loaded": encoder is not None,
            "scaler_loaded": scaler is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "model_loaded": False
        }

@app.get("/preprocessing-info")
def preprocessing_info():
    """Get information about available preprocessing steps"""
    return {
        "preprocessing_pipeline": {
            "step_1_encoding": {
                "available": encoder is not None,
                "encoder_type": type(encoder).__name__ if encoder else None,
                "encoder_file": encoder_filename if encoder is not None else None
            },
            "step_2_scaling": {
                "available": scaler is not None,
                "scaler_type": type(scaler).__name__ if scaler else None,
                "scaler_file": scaler_filename if scaler is not None else None
            }
        },
        "usage_note": "Send raw features - preprocessing will be applied automatically"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make predictions using the loaded model with automatic preprocessing"""
    try:
        # Validate input
        if not request.features:
            raise HTTPException(status_code=400, detail="Features list cannot be empty")
        
        logger.info(f"Raw input features: {request.features}")
        
        # Apply preprocessing pipeline
        processed_features = preprocess_features(request.features)
        
        logger.info(f"Processed features shape: {processed_features.shape}")
        
        # Make prediction
        prediction = model.predict(processed_features)
        
        # Convert numpy types to Python native types for JSON serialization
        if hasattr(prediction, 'tolist'):
            prediction_list = prediction.tolist()
        else:
            prediction_list = list(prediction)
        
        return PredictionResponse(
            prediction=prediction_list,
            model_info=f"Model: {type(model).__name__}",
            preprocessing_applied={
                "encoding": encoder is not None,
                "scaling": scaler is not None
            },
            input_shape=list(processed_features.shape)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input features: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions with automatic preprocessing"""
    try:
        # Validate input
        if not request.batch_features:
            raise HTTPException(status_code=400, detail="Batch features cannot be empty")
        
        logger.info(f"Batch input shape: {len(request.batch_features)}x{len(request.batch_features[0]) if request.batch_features else 0}")
        
        # Apply preprocessing pipeline to batch
        processed_features = preprocess_batch_features(request.batch_features)
        
        logger.info(f"Processed batch features shape: {processed_features.shape}")
        
        # Make batch prediction
        predictions = model.predict(processed_features)
        
        # Convert to list format
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
            # Ensure it's a list of lists for batch response
            if not isinstance(predictions_list[0], list):
                predictions_list = [[pred] for pred in predictions_list]
        else:
            predictions_list = [[pred] for pred in predictions]
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            model_info=f"Model: {type(model).__name__}",
            preprocessing_applied={
                "encoding": encoder is not None,
                "scaling": scaler is not None
            },
            batch_size=len(request.batch_features)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid batch features: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI ML Model Server with Preprocessing...")
    print(f"📊 Model: {model_filename}")
    print(f"🔧 Encoder: {encoder_filename}")
    print(f"📏 Scaler: {scaler_filename}")
    print(f"🌐 Server will run on: http://localhost:8000")
    print(f"📖 API Documentation: http://localhost:8000/docs")
    print(f"⚡ Interactive API: http://localhost:8000/redoc")
    print(f"🔍 Preprocessing Info: http://localhost:8000/preprocessing-info")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
