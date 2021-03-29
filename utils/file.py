import numpy as np
import pandas as pd
import os
import tqdm
import torch

from itertools import chain

import matplotlib.pyplot as plt

from PIL import Image

def img_loader(path: str):
    with Image.open(path) as img:
        img = np.array(img)
    return img

def read_all_png_in_dirs(base_paths : list, limit : int = None) -> (str, np.array):
    iters = [read_all_png_in_dir_iter(p, limit) for p in base_paths]
    return chain.from_iterable(iters)

def read_all_png_in_dir_iter(path2dir : str, limit : int = None) -> (str, np.array):
    cnt = 0
    for i, file in enumerate(os.listdir(path2dir)):
        if '.png' == file[-4:].lower():
            cnt += 1
            if limit is not None and cnt > limit:
                return
            img_path = os.path.join(path2dir, file)
            yield file, img_loader(img_path)

def read_all_png_in_dir(base_path : str, limit : int = None) -> dict:
    from os import listdir
    from os.path import isfile, join
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import tqdm

    imgs = [os.path.join(path, name) for path, subdirs, files in os.walk(base_path) for name in files]
    imgs = filter(lambda f: isfile(f) and '.png' == f[-4:], imgs)
    imgs = list(imgs)
    if limit is not None:
        imgs = imgs[:limit]
    data = {img : mpimg.imread(img) for img in tqdm.tqdm(imgs, position=0)}
    return data
