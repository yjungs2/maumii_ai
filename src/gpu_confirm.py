import torch

if torch.cuda.is_available():
    print("GPU 사용 가능:", torch.cuda.get_device_name(0))
    print("총 메모리:", round(torch.cuda.get_device_properties(0).total_memory/1024**3,2), "GB")
    print("CUDA 버전:", torch.version.cuda)
else:
    print("GPU 없음, CPU 사용")