import numpy as np
import pandas as pd
import os
import torch

from itertools import chain

from PIL import Image
def img_loader(path: str):
    with Image.open(path) as img:
        img = np.array(img)
    return img

def read_all_png_in_test_dir(base_path : str, limit : int = None) -> dict:
    from os import listdir
    from os.path import isfile, join
    data = {}
    path2dir = os.path.join(base_path, 'public_test')
    for file in os.listdir(path2dir):
        if '.png' == file[-4:]:
            img = os.path.join(path2dir, file)
            data[img] = img_loader(img)
    path2dir = os.path.join(base_path, 'private_test')
    for file in os.listdir(path2dir):
        if '.png' == file[-4:]:
            img = os.path.join(path2dir, file)
            data[img] = img_loader(img)
    return data
