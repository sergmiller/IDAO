import numpy as np
import pandas as pd
import os
import tqdm
import torch

from itertools import chain

import matplotlib.pyplot as plt


from .dataset import LabeledDataset
from .applier import build_embd_dataset


def apply_all_models_to_test_dataset(
    d : LabeledDataset,
    model1,
    model2,
    key : str,
    _emb_cache={}
) -> LabeledDataset:
    if key not in _emb_cache:
        _emb_cache[key] = build_embd_dataset(d)
    emb_dataset = _emb_cache.get(key)
 
    labels1 = model1.predict(emb_dataset.samples)
    labels2 = model2.predict(emb_dataset.samples)
    
    labels = np.stack([labels1, labels2]).T
    
    emb_dataset.labels = labels
    
    return emb_dataset