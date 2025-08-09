import os

def generate_fastapi_app(model_path: str, output_dir: str, port: int = 8000):
    """Generate FastAPI app with embedded template to avoid template directory dependency"""
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Embedded FastAPI template to avoid external template file dependency
    fastapi_template = '''from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List, Union
import os

# Load the trained model
try:
    model = joblib.load("{{ model_path }}")
    print(f"✅ Model loaded successfully from: {{ model_path }}")
except Exception as e:
    print(f"❌ Error loading model from {{ model_path }}: {e}")
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
        "model_path": "{{ model_path }}",
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
    print(f"📊 Model: {{ model_path }}")
    print(f"🌐 Server will run on: http://localhost:{{ port }}")
    print(f"📖 API Documentation: http://localhost:{{ port }}/docs")
    print(f"⚡ Interactive API: http://localhost:{{ port }}/redoc")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port={{ port }},
        log_level="info"
    )
'''
    
    # Render the template by replacing placeholders
    rendered_code = fastapi_template.replace("{{ model_path }}", model_path).replace("{{ port }}", str(port))
    
    # Save main.py
    main_py_path = os.path.join(output_dir, "main.py")
    with open(main_py_path, "w", encoding="utf-8") as f:
        f.write(rendered_code)
    
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
    
    # Create a simple README for the deployment
    readme_content = f"""# FastAPI ML Model Deployment

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

## API Endpoints

- `GET /` - Health check and basic info
- `GET /health` - Detailed health check
- `POST /predict` - Make predictions

## Example Usage

### Using curl:
```bash
# Health check
curl http://localhost:{port}/

# Make prediction (adjust features as needed)
curl -X POST "http://localhost:{port}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"features": [1.0, 2.0, 3.0, 4.0]}}'
```

### Using Python requests:
```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:{port}/predict",
    json={{"features": [1.0, 2.0, 3.0, 4.0]}}
)
print(response.json())
```

## Model Information
- **Model file:** {model_path}
- **Port:** {port}
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create a simple Docker file for containerization
    dockerfile_content = f"""FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY main.py .
COPY {os.path.basename(model_path)} .

# Expose the port
EXPOSE {port}

# Run the application
CMD ["python", "main.py"]
"""
    
    dockerfile_path = os.path.join(output_dir, "Dockerfile")
    with open(dockerfile_path, "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    
    print(f"📄 Created FastAPI app: {main_py_path}")
    print(f"📦 Created requirements.txt: {requirements_path}")
    print(f"📋 Created README.md: {readme_path}")
    print(f"🐳 Created Dockerfile: {dockerfile_path}")

# Import datetime for README timestamp
from datetime import datetime