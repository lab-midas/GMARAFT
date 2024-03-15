import json
import numpy as np

def crop_center(img, cropx, cropy):
    if len(img.shape) == 2:
        x, y = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[startx:startx + cropx, starty:starty + cropy]
    elif len(img.shape) == 4:
        _, _, x, y = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[..., startx:startx + cropx, starty:starty + cropy]
    elif len(img.shape) == 3:
        _, x, y = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[:, startx:startx + cropx, starty:starty + cropy]

