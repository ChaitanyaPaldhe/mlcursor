import subprocess
import sys
import re
import os

# Mapping of Python package names to pip package names
PACKAGE_MAPPINGS = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "xgb": "xgboost",
    "lgb": "lightgbm"
}

def ensure_script_dependencies(script_path: str):
    """Parse a Python script and install all missing imports."""
    # Fix: Use UTF-8 encoding explicitly
    with open(script_path, "r", encoding='utf-8') as f:
        code = f.read()
    
    # Find import statements
    imports = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", code, re.MULTILINE)
    
    # Remove duplicates and standard library modules
    stdlib_modules = {
        "os", "sys", "re", "json", "time", "datetime", "math", "random",
        "collections", "itertools", "functools", "pathlib", "uuid", "io",
        "subprocess", "threading", "multiprocessing", "logging", "typing"
    }
    
    external_packages = set()
    for pkg in imports:
        if pkg not in stdlib_modules:
            external_packages.add(pkg)  # Keep the original import name
    
    if not external_packages:
        print("✓ No external dependencies to install")
        return
    
    print(f"📦 Checking dependencies: {', '.join(external_packages)}")
    
    # Check which packages are missing
    missing_packages = []
    for package in external_packages:
        try:
            __import__(package)  # Try to import with the actual Python import name
        except ImportError:
            # Map to pip package name if needed
            pip_name = PACKAGE_MAPPINGS.get(package, package)
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"🔧 Installing missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--quiet"
                ])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install {package}: {e}")
    else:
        print("✅ All dependencies are already installed")

def install_package(package_name: str) -> bool:
    """Install a single package using pip"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name, "--quiet"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package_installed(package_name: str) -> bool:
    """Check if a package is already installed"""
    try:
        __import__(package_name)  # Use the actual import name, not pip name
        return True
    except ImportError:
        return False