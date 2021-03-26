import numpy as np
import pandas as pd
import os
import torch

from itertools import chain
from os import listdir
from os.path import join

from PIL import Image

def _fix_tag(tag) -> str:
    f = lambda x : '.'.join(x.split('/')[-1].split('.')[:-1]) # a/bc/d.efg.hi -> d.efg
    return f(tag)

def img_loader(path: str):
    with Image.open(path) as img:
        img = np.array(img)
    return img

def test_dir_png_reader(base_path : str, limit : int = None) -> (str, 'np.array'):
    path2dir = os.path.join(base_path, 'public_test')
    for file in os.listdir(path2dir):
        if '.png' == file[-4:].lower():
            img_path = os.path.join(path2dir, file)
            img_name = _fix_tag(file)
            yield img_name, img_loader(img_path)
    path2dir = os.path.join(base_path, 'private_test')
    for file in os.listdir(path2dir):
        if '.png' == file[-4:].lower():
            img_path = os.path.join(path2dir, file)
            img_name = _fix_tag(file)
            yield img_name, img_loader(img_path)
