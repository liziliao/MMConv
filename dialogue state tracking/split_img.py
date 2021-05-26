import os
import random
import numpy as np
from utils.json_utils import load

img_classes = load('./image_classes.json')                                                                                # print(img_classes)
img2class = {v: k for k, vs in img_classes.items() for v in vs}

img_root = ''
all_imgs = {file: os.path.join(root, file) for root, _, files in os.walk(os.path.join(img_root, 'data/images')) for file in files if file.lower().endswith(('.jpg', '.png', '.jpeg'))}


img_paths = list(set(all_imgs.values()).intersection(img2class.keys()))
random.shuffle(img_paths)
nof_imgs = len(img_paths)

splits = {
    'test': 0.1,
    'val': 0.05,
    'train': 0.85
}
assert sum(splits.values()) == 1

begin = 0
for i, (split, ratio) in enumerate(splits.items()):
    if i == len(splits) - 1:
        end = nof_imgs
    else:
        end = round(begin + ratio * nof_imgs)
    with open(f'resources/{split}.image', 'w+') as f:
        for j, path in enumerate(img_paths[begin: end]):
            if j == end - begin - 1:
                f.write(path)
            else:
                f.write(f'{path}\n')
    print(split, end - begin)
    begin = end
