import numpy as np
import pandas as pd
import os
import tqdm
import torch

from itertools import chain

def process_train_sample(item):
    assert 2 == len(item)

    tag = item[0]
    sample = item[1]

    assert isinstance(sample, np.ndarray)
    assert (576, 576) == sample.shape

    sample = np.array(255 * sample, dtype=np.uint8)

    is_NR = tag.find('_NR_') != -1

    tag_parts = tag.split('_')
    kev_part_id = [i for i, part in enumerate(tag_parts) if part == 'keV']
    assert len(kev_part_id) == 1 and kev_part_id[0] > 0
    kev_part_id = kev_part_id[0]

    level = int(tag_parts[kev_part_id - 1])

    label = np.array(["NR" if is_NR else "ER", level])

    return tag, sample, label


def process_test_sample(item):
    assert 2 == len(item)

    tag = item[0]
    sample = item[1]

    assert isinstance(sample, np.ndarray)
    assert (576, 576) == sample.shape

    sample = np.array(255 * sample, dtype=np.uint8)

    return tag, sample, []
