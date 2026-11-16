import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

modules = [
    "cv2", "numpy", "matplotlib", "seaborn",
    "sklearn", "tensorflow", "streamlit", "pytest", "nbformat"
]

for mod in modules:
    try:
        m = __import__(mod)
        print(f"{mod}: {getattr(m, '__version__', 'available')}")
    except ImportError:
        print(f"{mod}: NOT INSTALLED")
