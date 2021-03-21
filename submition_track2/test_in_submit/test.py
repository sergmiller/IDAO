import numpy as np
import torch

mobilenet = torch.load("mobilenet")

np.random.seed(0)
t = np.random.random((1,3,500,500))
model2.train(False)
print(np.mean(t), np.std(t))
with torch.no_grad():
    res = mobilenet(torch.Tensor(t)).detach().numpy()
print(np.mean(res), np.std(res))



import time
start = time.time()
for i in np.arange(10000):
    t = np.random.random((1,3,500,500))
    res = mobilenet(torch.Tensor(t)).detach().numpy()
    
print('OK, run {}s'.format(time.time() - start))
