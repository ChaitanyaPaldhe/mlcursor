# === File: core/deps.py ===
import importlib.util
import subprocess
import sys
import re

def is_installed(pkg: str) -> bool:
    """Check if a package is installed."""
    return importlib.util.find_spec(pkg) is not None

def install(pkg: str):
    """Install the given package via pip."""
    print(f"📦 Installing missing dependency: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def ensure_script_dependencies(script_path: str):
    """Parse a Python script and install all missing imports."""
    with open(script_path, "r") as f:
        code = f.read()

    # Find import statements
    imports = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z0-9_]+)", code, re.MULTILINE)

    # Remove common standard libraries (optional)
    stdlib_exclude = {
        "os", "sys", "re", "json", "uuid", "time", "subprocess",
        "typing", "math", "random", "datetime", "collections", "itertools"
    }

    unique_imports = sorted(set(imports) - stdlib_exclude)

    for pkg in unique_imports:
        if not is_installed(pkg):
            install(pkg)
        else:
            print(f"✅ {pkg} already installed.")
