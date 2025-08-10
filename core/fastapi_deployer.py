import os
import glob
from pathlib import Path

def generate_fastapi_app(model_path: str, output_dir: str, port: int = 8000):
    """Generate FastAPI app with automatic encoder/scaler detection and preprocessing"""
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-detect encoder and scaler files in the same directory as the model
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Search patterns for encoder and scaler files
    encoder_patterns = [
        f"{model_name}_encoder.pkl",
        f"{model_name}_encoder.joblib",
        "*encoder*.pkl",
        "*encoder*.joblib"
    ]
    
    scaler_patterns = [
        f"{model_name}_scaler.pkl",
        f"{model_name}_scaler.joblib",
        "*scaler*.pkl",
        "*scaler*.joblib"
    ]
    
    # Find encoder and scaler files
    encoder_path = None
    scaler_path = None
    
    for pattern in encoder_patterns:
        matches = glob.glob(os.path.join(model_dir, pattern))
        if matches:
            encoder_path = matches[0]
            break
    
    for pattern in scaler_patterns:
        matches = glob.glob(os.path.join(model_dir, pattern))
        if matches:
            scaler_path = matches[0]
            break
    
    print(f"🔍 Model: {model_path}")
    print(f"🔍 Encoder: {encoder_path if encoder_path else 'Not found'}")
    print(f"🔍 Scaler: {scaler_path if scaler_path else 'Not found'}")
    
    # Generate FastAPI code with proper conditional logic
    fastapi_code = generate_fastapi_code(model_path, encoder_path, scaler_path, port)
    
    # Save main.py
    main_py_path = os.path.join(output_dir, "main.py")
    with open(main_py_path, "w", encoding="utf-8") as f:
        f.write(fastapi_code)
    
    # Enhanced requirements.txt with more dependencies
    requirements_content = """fastapi>=0.68.0
uvicorn[standard]>=0.15.0
scikit-learn>=1.0.0
joblib>=1.0.0
numpy>=1.21.0
pydantic>=1.8.0
python-multipart>=0.0.5
"""
    
    # Save requirements.txt
    requirements_path = os.path.join(output_dir, "requirements.txt")
    with open(requirements_path, "w") as f:
        f.write(requirements_content)
    
    # Copy model and preprocessing files to output directory
    import shutil
    
    # Copy model file
    model_dest = os.path.join(output_dir, os.path.basename(model_path))
    if not os.path.exists(model_dest):
        shutil.copy2(model_path, model_dest)
        print(f"📋 Copied model: {model_dest}")
    else:
        print(f"📋 Model already exists: {model_dest}")
    
    # Copy encoder file if exists
    if encoder_path:
        encoder_dest = os.path.join(output_dir, os.path.basename(encoder_path))
        if not os.path.exists(encoder_dest):
            shutil.copy2(encoder_path, encoder_dest)
            print(f"🔧 Copied encoder: {encoder_dest}")
        else:
            print(f"🔧 Encoder already exists: {encoder_dest}")
    
    # Copy scaler file if exists
    if scaler_path:
        scaler_dest = os.path.join(output_dir, os.path.basename(scaler_path))
        if not os.path.exists(scaler_dest):
            shutil.copy2(scaler_path, scaler_dest)
            print(f"📏 Copied scaler: {scaler_dest}")
        else:
            print(f"📏 Scaler already exists: {scaler_dest}")
    
    # Create README and Dockerfile...
    create_additional_files(output_dir, model_path, encoder_path, scaler_path, port)
    
    print(f"📄 Created FastAPI app: {main_py_path}")
    print(f"📦 Created requirements.txt: {requirements_path}")
    print(f"✨ Auto-preprocessing: {'✅ Enabled' if (encoder_path or scaler_path) else '❌ No preprocessing files found'}")
    print(f"\n🚀 To run the API, execute:")
    print(f"   cd {output_dir}")
    print(f"   python main.py")

def generate_fastapi_code(model_path: str, encoder_path: str, scaler_path: str, port: int) -> str:
    """Generate FastAPI code with proper conditional logic"""
    
    # Header imports and logging setup
    code = '''from fastapi import FastAPI, HTTPException
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
model_filename = "''' + os.path.basename(model_path) + '''"
model_path = os.path.join(SCRIPT_DIR, model_filename)

try:
    model = joblib.load(model_path)
    logger.info(f"✅ Model loaded successfully from: {model_path}")
except Exception as e:
    logger.error(f"❌ Error loading model from {model_path}: {e}")
    raise

'''
    
    # Conditional encoder loading
    if encoder_path:
        code += '''# Load encoder
encoder = None
encoder_filename = "''' + os.path.basename(encoder_path) + '''"
encoder_path = os.path.join(SCRIPT_DIR, encoder_filename)

try:
    encoder = joblib.load(encoder_path)
    logger.info(f"✅ Encoder loaded successfully from: {encoder_path}")
except Exception as e:
    logger.warning(f"⚠️ Could not load encoder from {encoder_path}: {e}")
    encoder = None

'''
    else:
        code += '''# No encoder available
encoder = None

'''
    
    # Conditional scaler loading
    if scaler_path:
        code += '''# Load scaler
scaler = None
scaler_filename = "''' + os.path.basename(scaler_path) + '''"
scaler_path = os.path.join(SCRIPT_DIR, scaler_filename)

try:
    scaler = joblib.load(scaler_path)
    logger.info(f"✅ Scaler loaded successfully from: {scaler_path}")
except Exception as e:
    logger.warning(f"⚠️ Could not load scaler from {scaler_path}: {e}")
    scaler = None

'''
    else:
        code += '''# No scaler available
scaler = None

'''
    
    # FastAPI app and models
    code += '''app = FastAPI(
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
    print(f"📊 Model: {model_filename}")'''
    
    if encoder_path:
        code += '''
    print(f"🔧 Encoder: {encoder_filename}")'''
    
    if scaler_path:
        code += '''
    print(f"📏 Scaler: {scaler_filename}")'''
    
    code += '''
    print(f"🌐 Server will run on: http://localhost:''' + str(port) + '''")
    print(f"📖 API Documentation: http://localhost:''' + str(port) + '''/docs")
    print(f"⚡ Interactive API: http://localhost:''' + str(port) + '''/redoc")
    print(f"🔍 Preprocessing Info: http://localhost:''' + str(port) + '''/preprocessing-info")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=''' + str(port) + ''',
        log_level="info"
    )
'''
    
    return code

def create_additional_files(output_dir: str, model_path: str, encoder_path: str, scaler_path: str, port: int):
    """Create README.md and Dockerfile"""
    
    # Create README
    preprocessing_info = ""
    if encoder_path or scaler_path:
        preprocessing_info = f"""
## Automatic Preprocessing

This API automatically applies preprocessing to your input features:

{"1. **Encoding**: Categorical features are encoded using the trained encoder" if encoder_path else ""}
{"2. **Scaling**: Numeric features are scaled using the trained scaler" if scaler_path else ""}

**Important**: Send raw, unprocessed features to the API. The preprocessing will be applied automatically.

### Preprocessing Files:
{f"- Encoder: `{os.path.basename(encoder_path)}`" if encoder_path else ""}
{f"- Scaler: `{os.path.basename(scaler_path)}`" if scaler_path else ""}
"""
    
    readme_content = f"""# FastAPI ML Model Deployment with Preprocessing

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
   - Main endpoint: http://localhost:{port}
   - API Documentation: http://localhost:{port}/docs
   - Alternative docs: http://localhost:{port}/redoc
   - Preprocessing info: http://localhost:{port}/preprocessing-info

{preprocessing_info}

## Model Information
- **Model file:** {os.path.basename(model_path)}
{f"- **Encoder file:** {os.path.basename(encoder_path)}" if encoder_path else ""}
{f"- **Scaler file:** {os.path.basename(scaler_path)}" if scaler_path else ""}
- **Port:** {port}
- **Auto-preprocessing:** {"✅ Enabled" if (encoder_path or scaler_path) else "❌ Disabled"}

## Example Usage

### Single Prediction
```bash
curl -X POST "http://localhost:{port}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"features": [5.1, 3.5, 1.4, 0.2]}}'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:{port}/predict/batch" \\
     -H "Content-Type: application/json" \\
     -d '{{"batch_features": [[5.1, 3.5, 1.4, 0.2], [6.0, 3.0, 4.0, 1.2]]}}'
```
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create Dockerfile
    dockerfile_content = f"""FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and model files
COPY main.py .
COPY {os.path.basename(model_path)} .
{f"COPY {os.path.basename(encoder_path)} ." if encoder_path else ""}
{f"COPY {os.path.basename(scaler_path)} ." if scaler_path else ""}

# Expose the port
EXPOSE {port}

# Run the application
CMD ["python", "main.py"]
"""
    
    dockerfile_path = os.path.join(output_dir, "Dockerfile")
    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    
    print(f"📋 Created README.md: {readme_path}")
    print(f"🐳 Created Dockerfile: {dockerfile_path}")

# Import datetime for README timestamp
from datetime import datetime

# Example usage
if __name__ == "__main__":
    # Example call - adjust the paths according to your setup
    model_path = "models/logistic_regression_iris.pkl"  # Adjust this path
    output_dir = "deployment_fastapi"
    port = 8000
    
    generate_fastapi_app(model_path, output_dir, port)