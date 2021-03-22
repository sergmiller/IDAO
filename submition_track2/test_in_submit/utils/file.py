import numpy as np
import pandas as pd
import os
import torch

from itertools import chain

# import matplotlib.pyplot as plt

from PIL import Image
def img_loader(path: str):
    with Image.open(path) as img:
        img = np.array(img)
    return img

def read_all_png_in_dir(base_path : str, limit : int = None) -> dict:
    from os import listdir
    from os.path import isfile, join
    # import matplotlib.pyplot as plt

    imgs = [os.path.join(dir,f) for (dir, subdirs, fs) in os.walk(base_path) for f in fs]
    # data = {}
    # for file in os.listdir('tests/public_test'):
    #     if '.png' == file[-4:]:
    #         img = os.path.join('tests/public_test', file)
    #         data[img] = img_loader(img)
    # for file in os.listdir('tests/private_test'):
    #     if '.png' == file[-4:]:
    #         img = os.path.join('tests/public_test', file)
    #         data[img] = img_loader(img)
    print('len of imgs', len(imgs))
    imgs = filter(lambda f: isfile(f) and '.png' == f[-4:], imgs)
    imgs = list(imgs)
    if limit is not None:
        imgs = imgs[:limit]
    data = {img : img_loader(img) for img in imgs}
    return data
