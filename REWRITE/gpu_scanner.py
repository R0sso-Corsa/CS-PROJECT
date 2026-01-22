import torch

print("="*60)
print("PYTORCH DIAGNOSTIC")
print("="*60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"PyTorch Built With CUDA: {torch.version.cuda}")
print(f"PyTorch Built With ROCm: {torch.version.hip}")

# For NVIDIA (won't work for AMD)
print(f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")

# Try to get device name (works for both NVIDIA and AMD with ROCm)
try:
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
except:
    print("No GPU detected via CUDA API")

print("\n" + "="*60)