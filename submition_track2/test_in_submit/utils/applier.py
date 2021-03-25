import numpy as np
import pandas as pd
import os
import tqdm
import torch

from itertools import chain

from joblib import Parallel, delayed

import torch

from .dataset import LabeledDataset

import torchvision.models as models
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
try:
    mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
except:
    mobilenet_v3_small = None
    print("Can't init pretrained model")

def build_embd_dataset(
    d : LabeledDataset,
    model=mobilenet_v3_small,
    batch_size : int=16
) -> LabeledDataset:
    assert model is not None
    pretrained_embeds = []

    model.train(False)

    def f(i):
        batch = d.samples[i:i+batch_size]
        batch = torch.FloatTensor(batch.reshape(-1,1,576,576)).repeat(1,3,1,1) / 255
        with torch.no_grad():
            emb = model(batch).detach().numpy()
        assert emb.shape[0] > 0
        return emb


    with Parallel(n_jobs=7) as parallel:
        pretrained_embeds = parallel(delayed(f)(i) for i in tqdm.tqdm(
            np.arange(0, d.samples.shape[0], batch_size), position=0))

    pretrained_embeds_flat = []

    for batch in pretrained_embeds:
        for x in batch:
            pretrained_embeds_flat.append(x)

    pretrained_embeds = np.array(pretrained_embeds_flat).reshape(-1, 1000)


    emb_dataset = LabeledDataset()
    emb_dataset.samples = pretrained_embeds
    emb_dataset.labels = d.labels
    emb_dataset.tags = d.tags

    return emb_dataset
