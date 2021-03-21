import numpy as np
import torch


import time
start = time.time()
for i in np.arange(100):
    t = np.random.random((32,3,500,500))
    with torch.no_grad():
        res = mobilenet(torch.Tensor(t)).detach().numpy()
