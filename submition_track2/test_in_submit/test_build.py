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

import numpy as np
import torch
import torchvision.transforms as transforms
import os

import sys

# sys.path.append('..')

import utils

from importlib import reload

reload(utils)

from utils import dataset, applier, domain, pipe, file

reload(file)
reload(dataset)
reload(applier)
reload(domain)
reload(pipe)

from utils.dataset import LabeledDataset
from utils.file import read_all_png_in_dirs, read_all_png_in_dir_iter


from mobilenet_v3 import mobilenet_v3_small

mobilenet = mobilenet_v3_small()
mobilenet.load_state_dict(torch.load("models/mobilenet_state_dict"))

croped_mobilenet = torch.nn.Sequential(
    transforms.CenterCrop(64),
    applier.mobilenet_v3_small
)

def apply_random_projection_and_normalize(data : np.array, to_dim : int = 10, seed : int = 0):
    assert 2 == len(data.shape)
    np.random.seed(seed)
    N = data.shape[1]
    proj_matrix = np.random.randn(N, to_dim) / np.sqrt(N)
#     print(np.mean(proj_matrix), np.std(proj_matrix))
    data_proj = np.matmul(data, proj_matrix)
    data_proj /= np.sum(np.abs(data_proj), axis=1, keepdims=True)
    return data_proj


def mae_scorer(estimator, X, y):
    pass  # need this declaration to upload model from pickle


class ModelWithProjection:
    def __init__(self, model):
        self.model = model
    def predict(self, x):
        x_proj = apply_random_projection_and_normalize(x, to_dim=64)
        return self.model.predict(x_proj)
    def predict_proba(self, x):
        x_proj = apply_random_projection_and_normalize(x, to_dim=64)
        return self.model.predict_proba(x_proj)
        self.model.train(False)

import pickle

with open('models/cv_label1.pkl', 'rb') as f:
    cv_label1 = pickle.load(f)

with open('models/cv_label2.pkl', 'rb') as f:
    cv_label2 = pickle.load(f)

model1 = ModelWithProjection(cv_label1)
model2 = ModelWithProjection(cv_label2)


def pipe_for_dir(dir : dir) -> dataset.LabeledDataset:
    return pipe.apply_all_models_to_test_dataset_gen(
        dataset.build_batched_dataset(
            [dir],
            domain.process_test_sample,
            limit=128,
            batch_size=16),
        croped_mobilenet,
        model1,
        model2
    )



import pandas as pd
from utils import file
# PATH_TO_TEST_DIRS = "../../data/idao_dataset"
PATH_TO_TEST_DIRS = os.path.abspath('./tests')
# d1 = file.read_all_png_in_test_dir(PATH_TO_TEST_DIRS)

print('Build ok')
