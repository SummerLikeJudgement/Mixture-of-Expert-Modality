import torch

# 检查 CUDA 是否可用
print(torch.cuda.is_available())  # 返回 True 表示 CUDA 可用

# 检查当前 CUDA 设备数量
print(torch.cuda.device_count())  # 返回可用 GPU 数量

if torch.cuda.device_count():
    # 检查当前 CUDA 设备名称
    print(torch.cuda.get_device_name(0))  # 返回 GPU 型号（如 "NVIDIA GeForce RTX 3090"）

# 检查 CUDA 版本（PyTorch 编译时使用的 CUDA 版本）
print(torch.version.cuda)  # 如 "12.1"