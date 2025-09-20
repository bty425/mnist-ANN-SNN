import torch

cuda_available = torch.cuda.is_available()
print("PyTorch 版本:", torch.__version__)
print(f" CUDA 是否可用: {cuda_available}")
