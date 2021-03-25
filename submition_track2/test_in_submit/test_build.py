import numpy as np
import torch
from mobilenet_v3 import mobilenet_v3_small

import os
print('Is your solution okie on private test?')
# from utils import file, dataset
# PATH_TO_TEST_DIRS = os.path.abspath('./tests')
# d1 = file.read_all_png_in_test_dir(PATH_TO_TEST_DIRS)
# print('len d1', len(d1))
# mobilenet = mobilenet_v3_small()
# mobilenet.load_state_dict(torch.load("mobilenet_state_dict"))
#
# np.random.seed(0)
# t = np.random.random((4,3,32,32))
# mobilenet.train(False)
# print(np.mean(t), np.std(t))
# with torch.no_grad():
#     res = mobilenet(torch.Tensor(t)).detach().numpy()
# print(np.mean(res), np.std(res))
