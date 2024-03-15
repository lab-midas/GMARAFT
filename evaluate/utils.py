import numpy as np

def get_data(img, slice, t1, t2, mode):
    if mode == 'pairwise':
        idx1, idx2 = get_neighboring_frames_pairwise(25, t2)
        ref, mov = img_preprocessing(img[slice, t1], scale=1), img_preprocessing(img[slice, t2], scale=1)
        context = np.stack((img_preprocessing(img[slice, idx1], scale=1),
                            img_preprocessing(img[slice, t2], scale=1),
                            img_preprocessing(img[slice, idx2], scale=1)), axis=0)
        return ref, mov, context


def increase_brightness(img, value=0.5):
    max = np.max(img)
    value = max * value
    lim = max - value
    img[img > lim] = max
    img[img <= lim] += value
    return img


def add_quiver(ax, u, stride=8, offset=None, scale=40, **kwargs):
    if offset is None:
        offset = [0, 0]
    x, y = u.shape[:2]
    gridx, gridy = np.meshgrid(np.arange(0, y, stride), np.arange(0, x, stride))
    gridx, gridy = gridx + offset[0], gridy + offset[1]
    # to make it consistent with color wheel coordinate (and also many others), uy should be negative here
    ax.quiver(gridx, gridy, u[0:x:stride, :, :][:, 0:y:stride, :][:, :, 0],
              -u[0:x:stride, :, :][:, 0:y:stride, :][:, :, 1], color='y', scale_units='inches', scale=scale,
              headwidth=5)

def get_neighboring_frames_pairwise(n_frames, t):
        idx1, idx2 = t - 1, t + 1
        if t == 0:
            idx1 = 24
        if t == (n_frames - 1):
            idx2 = 0
        return idx1, idx2


def img_preprocessing(img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img
