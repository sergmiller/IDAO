import sys, os

# Disable
def blockPrint():
    sys.stderr = open(os.devnull, 'w')
    sys.stdout = open(os.devnull, 'w')

blockPrint()

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
    mobilenet
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


class TrainLightModel:
    def __init__(self, energy_model, particle_model):
        self.energy_model = energy_model
        self.particle_model = particle_model

    def predict_particle(self, x):
        x = apply_random_projection_and_normalize(x, to_dim=64)
        return self.particle_model.predict_proba(x)[:, 0]

    def predict_energy(self, x):
        x = apply_random_projection_and_normalize(x, to_dim=64)
        return self.energy_model.predict(x)


class TestLightModel:
    # X is for 1 and 6, also 30 have 2 clusters
    CLUSTER_TO_ENERGY = ['X', '20', '30', '3', '10', '30']
    CLUSTER_1_6_MEDIAN = 0.8917
    CLASSES = ['1', '10', '20', '3', '30', '6']  # alphabet order
    CLASS2ID = {c:i for i,c in enumerate(CLASSES)}
    def __init__(self, energy_model, cluster_model):
        self.energy_model = energy_model
        self.cluster_model = cluster_model

    def predict_particle(self, x):
        x = apply_random_projection_and_normalize(x, to_dim=64)
        probs = self.energy_model.predict_proba(x)
        probs_zero = probs[:, self.CLASS2ID['1']] \
            + probs[:, self.CLASS2ID['6']] + probs[:, self.CLASS2ID['20']]
        return probs_zero

    def predict_energy(self, x):
        x = apply_random_projection_and_normalize(x, to_dim=64)
        clusters = self.cluster_model.predict(x)
        probs = self.energy_model.predict_proba(x)[:, 0]  # 0 is for '1' energy
        is_energy_class_1 = probs > self.CLUSTER_1_6_MEDIAN
        result = np.array([self.CLUSTER_TO_ENERGY[x] for x in clusters])
        for i, c in enumerate(result):
            if c != "X":
                continue
            result[i] = '1' if is_energy_class_1[i] else '6'
        return result


import pickle

with open('models/cv_label1_mobilenet_crop_64.pkl', 'rb') as f:
    cv_label1 = pickle.load(f)

with open('models/cv_label2_mobilenet_crop_64.pkl', 'rb') as f:
    cv_label2 = pickle.load(f)

with open('models/cv_label2_tsne_test_v4.pkl', 'rb') as f:
    cv_cluster_model = pickle.load(f)


train_model = TrainLightModel(cv_label2, cv_label1)
test_model = TestLightModel(cv_label2, cv_cluster_model)


def pipe_for_dir(dir : dir, light_model) -> dataset.LabeledDataset:
    return pipe.apply_all_models_to_test_dataset_gen(
        dataset.build_batched_dataset(
            [dir],
            domain.process_test_sample,
            # limit=256,
            batch_size=48),
        croped_mobilenet,
        light_model
    )



import pandas as pd
from utils import file
# PATH_TO_TEST_DIRS = "../../data/idao_dataset"
PATH_TO_TEST_DIRS = os.path.abspath('./tests')

public_predictions = pipe_for_dir(os.path.join(PATH_TO_TEST_DIRS, 'public_test'), train_model)
private_predictions = pipe_for_dir(os.path.join(PATH_TO_TEST_DIRS, 'private_test'), test_model)

submit = dataset.LabeledDataset.merge(public_predictions, private_predictions)
# print(public_predictions.labels)
# exit(0)

dataset.dataset2submit_csv(submit, 'submission.csv')
#
# result2save = []
#
# loader = file.read_all_png_in_dirs(
#     [
#         os.path.join(PATH_TO_TEST_DIRS, 'public_test'),
#         os.path.join(PATH_TO_TEST_DIRS, 'private_test'),
#     ]
# )
#
# for key, array in loader:
#     result2save.append([key, 0, 0])
# result2save = np.array(result2save)
# result2save[:, 0] = dataset._fix_tags(result2save[:, 0])
# data_frame = pd.DataFrame(result2save, columns=["id", "classification_predictions", "regression_predictions"])
# data_frame.to_csv('submission.csv', index=False, header=True)

# print("OK")
