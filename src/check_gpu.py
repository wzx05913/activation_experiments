"""Print CUDA/PyTorch status for the active project interpreter."""
import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda_build:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
    print("gpu_memory_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
