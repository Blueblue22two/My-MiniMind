import torch

def print_torch_info():
    # 打印 PyTorch 版本
    print(f"PyTorch version: {torch.__version__}")
    
    # 检查 CUDA（NVIDIA GPU）
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  CUDA device {i}: {torch.cuda.get_device_name(i)}")
        current_cuda = torch.cuda.current_device()
        print(f"Current CUDA device: {current_cuda} ({torch.cuda.get_device_name(current_cuda)})")
    else:
        print("CUDA available: False")
    
    # 检查 MPS（Apple Silicon GPU）
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) available: True")
    else:
        print("MPS available: False")
    
    # 检查 CPU
    print(f"CPU available: True")
    
    # 推荐使用的设备
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"\nRecommended device: {device}")
    print(f"Current device tensor would use: {torch.tensor([1]).to(device).device}")

# 运行检测
if __name__ == "__main__":
    print_torch_info()

