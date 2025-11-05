#!/usr/bin/env python
import torch

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        print("Device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("Device name okunamadı:", e)
else:
    print("Device: CPU")
print("Torch version:", torch.__version__)
print("ROCm:", getattr(torch.version, 'hip', None))
