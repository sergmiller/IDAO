import numpy as np
import torch
import os

# from mobilenet_v3 import mobilenet_v3_small
#
# mobilenet = mobilenet_v3_small()
# mobilenet.load_state_dict(torch.load("mobilenet_state_dict"))



# import time
# start = time.time()
# for i in np.arange(1):
#     t = np.random.random((8,3,64,64))
#     with torch.no_grad():
#         res = mobilenet(torch.Tensor(t)).detach().numpy()

import pandas as pd
from utils import file
PATH_TO_TEST_DIRS = os.path.abspath('./tests')
# d1 = file.read_all_png_in_test_dir(PATH_TO_TEST_DIRS)
result2save = []
for key, array in file.test_dir_png_reader(PATH_TO_TEST_DIRS):
    result2save.append([key, 0, 0])
data_frame = pd.DataFrame(result2save, columns=["id", "classification_predictions", "regression_predictions"])
data_frame.to_csv('submission.csv', index=False, header=True)
