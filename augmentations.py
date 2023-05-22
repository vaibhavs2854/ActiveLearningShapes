import random
import numpy as np
from PIL import Image

def random_flip(input, axis, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            axis += 1
        return np.flip(input, axis=axis)
    else:
        return input


def random_crop(input, with_fa=False):
    
    ran = random.random()
    if ran > 0.2:
        # find a random place to be the left upper corner of the crop
        if with_fa:
            rx = int(random.random() * input.shape[1] // 10)
            ry = int(random.random() * input.shape[2] // 10)
            return input[:, rx: rx + int(input.shape[1] * 9 // 10), ry: ry + int(input.shape[2] * 9 // 10)]
        else:
            rx = int(random.random() * input.shape[0] // 10)
            ry = int(random.random() * input.shape[1] // 10)
            return input[rx: rx + int(input.shape[0] * 9 // 10), ry: ry + int(input.shape[1] * 9 // 10)]
    else:
        return input


def random_rotate_90(input, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            return np.rot90(input, axes=(1,2))
        return np.rot90(input)
    else:
        return input


def random_rotation(x, chance, with_fa=False):
    ran = random.random()
    if with_fa:
        img = Image.fromarray(x[0])
        mask = Image.fromarray(x[1])
        if ran > 1 - chance:
            # create black edges
            angle = np.random.randint(0, 90)
            img = img.rotate(angle=angle, expand=1)
            mask = mask.rotate(angle=angle, expand=1, fillcolor=1)
            return np.stack([np.asarray(img), np.asarray(mask)])
        else:
            return np.stack([np.asarray(img), np.asarray(mask)])
    img = Image.fromarray(x)
    if ran > 1 - chance:
        # create black edges
        angle = np.random.randint(0, 90)
        img = img.rotate(angle=angle, expand=1)
        return np.asarray(img)
    else:
        return np.asarray(img)