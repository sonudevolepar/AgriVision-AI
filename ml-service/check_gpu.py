
import torch
import torchvision

print("===================================")
print("       AgriVision AI GPU TEST")
print("===================================")

print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

    # GPU par tensor create karna
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")

    # GPU computation
    result = torch.matmul(x, y)

    print("GPU Computation: SUCCESS")
    print("Result Device:", result.device)

    memory = torch.cuda.get_device_properties(0).total_memory
    print("GPU Memory:", round(memory / (1024**3), 2), "GB")
else:
    print("GPU not available. CPU mode required.")

print("===================================")
