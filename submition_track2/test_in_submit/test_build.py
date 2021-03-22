import numpy as np
import torch
from mobilenet_v3 import mobilenet_v3_small

mobilenet = mobilenet_v3_small()
mobilenet.load_state_dict(torch.load("mobilenet_state_dict"))

print(torch.cuda.is_available())
print(torch.cuda.device_count())
np.random.seed(0)
t = np.random.random((4,3,32,32))
mobilenet.train(False)
print(np.mean(t), np.std(t))
with torch.no_grad():
    res = mobilenet(torch.Tensor(t)).detach().numpy()
print(np.mean(res), np.std(res))
import os
print(len(os.listdir('tests/public_test')))
print('example files', os.listdir('tests/public_test')[0:5])
