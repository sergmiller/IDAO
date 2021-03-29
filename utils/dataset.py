import numpy as np
import pandas as pd
import os
import tqdm
import torch

from itertools import chain

import matplotlib.pyplot as plt

from .file import read_all_png_in_dirs
from .domain import process_train_sample


class LabeledDataset:
    def __init__(self):
        self.samples = []
        self.labels = []
        self.tags = []

    def add_sample(self, tag : str, sample : np.array, label : np.array):
        self.samples.append(sample)
        self.labels.append(label)
        self.tags.append(tag)

    def finalize(self):
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        self.tags = np.array(self.tags)

    def save(self, file : str):
        np.savez_compressed(file, dataset=self)

    def subset(self, ids):
        d = LabeledDataset()
        d.samples = self.samples[ids]
        d.labels = self.labels[ids]
        d.tags = self.tags[ids]
        return d

    @staticmethod
    def load(file : str):
        return np.load(file, allow_pickle=True)['dataset'].item()

    @staticmethod
    def merge(a, b):
        c = LabeledDataset()
        c.samples = np.concatenate([a.samples, b.samples], axis=0)
        c.labels = np.concatenate([a.labels, b.labels], axis=0)
        c.tags = np.concatenate([a.tags, b.tags], axis=0)
        return c


def build_batched_dataset(paths : list, sample_processor=process_train_sample, limit : int = None, batch_size : int = None) -> LabeledDataset:
    samples = read_all_png_in_dirs(paths)
    cnt = 0
    d = LabeledDataset()
    for item in samples:
        cnt += 1
        tag, sample, label = sample_processor(item)
        d.add_sample(tag, sample, label)
        if limit is not None and cnt > limit:
            return
        if len(d.samples) == batch_size:
            d.finalize()
            yield d
            d = LabeledDataset()
    if len(d.samples) > 0:
        d.finalize()
        yield d

def build_dataset(paths : list, sample_processor=process_train_sample, limit : int = None) -> LabeledDataset:
    samples = read_all_png_in_dirs(paths, limit)
    d = LabeledDataset()
    for item in samples:
        tag, sample, label = sample_processor(item)
        d.add_sample(tag, sample, label)
    d.finalize()
    return d

def _fix_tags(tags) -> np.array:
    f = lambda x : '.'.join(x.split('/')[-1].split('.')[:-1]) # a/bc/d.efg.hi -> d.efg
    return np.array([f(tag) for tag in tags])

def dataset2submit_csv(d : LabeledDataset, fname : str):
    pred = np.stack([_fix_tags(d.tags), d.labels[:, 0], d.labels[:, 1]]).T
    data_frame = pd.DataFrame(
        pred,
        columns=["id", "classification_predictions", "regression_predictions"])
    data_frame.to_csv(fname, index=False, header=True)
