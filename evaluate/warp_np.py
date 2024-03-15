from skimage.transform import warp
import numpy as np

def warp_np(img, flow):
    flow = flow.astype('float32')
    height, width = np.shape(img)[0], np.shape(img)[1]
    posx, posy = np.mgrid[:height, :width]
    vx = flow[:, :, 1]
    vy = flow[:, :, 0]
    coord_x = posx + vx
    coord_y = posy + vy
    coords = np.array([coord_x, coord_y])
    img = img.astype('float32')
    warped = warp(img, coords, order=0)
    return warped