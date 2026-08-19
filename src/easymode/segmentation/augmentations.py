import numpy as np
import random
from scipy.ndimage import rotate, gaussian_filter
from skimage.transform import resize

from easymode.segmentation.membrain_fourier_augmentations.fourier_augmentations import MissingWedgeMaskAndFourierAmplitudeMatchingCombined

ROT_XZ_YZ_MAX_ANGLE = 15.0
ROT_XY_MAX_ANGLE = 22.5

def rotate_90_xy(img, label):
    k = random.randint(0, 4)
    img = np.rot90(img, k=k, axes=(1, 2))
    label = np.rot90(label, k=k, axes=(1, 2))
    return img, label

def rotate_90_xz(img, label):
    k = random.randint(0, 2) * 2
    img = np.rot90(img, k=k, axes=(0, 2))
    label = np.rot90(label, k=k, axes=(0, 2))
    return img, label

def flip(img, label):
    k = random.choice([0, 1, 2])
    img = np.flip(img, axis=k)
    label = np.flip(label, axis=k)
    return img, label

def rotate_continuous_xz_or_yz(img, label):
    plane = random.choice([(0, 2), (0, 1)])
    angle = np.random.uniform(-ROT_XZ_YZ_MAX_ANGLE, ROT_XZ_YZ_MAX_ANGLE)

    img = rotate(img, angle, axes=plane, order=1, mode='reflect', prefilter=False, reshape=False)
    label = rotate(label, angle, axes=plane, order=0, mode='constant', cval=2, reshape=False)

    return img, label

def rotate_continuous_xy(img, label):
    angle = np.random.uniform(-ROT_XY_MAX_ANGLE, ROT_XY_MAX_ANGLE)

    img = rotate(img, angle, axes=(1, 2), order=1, mode='reflect', prefilter=False, reshape=False)
    label = rotate(label, angle, axes=(1, 2), order=0, mode='constant', cval=2, reshape=False)

    return img, label

def remove_wedge(img, label):
    membrain_fourier_trickery_machine = MissingWedgeMaskAndFourierAmplitudeMatchingCombined()
    img = membrain_fourier_trickery_machine(img)
    return img, label

def filter_gaussian(img, label):
    img = gaussian_filter(img, sigma=random.uniform(0.5, 1.0))
    return img, label

def scale(img, label):
    factor = np.random.uniform(0.9, 1.1)
    shape = img.shape
    new_shape = tuple(int(round(s * factor)) for s in shape)

    zoomed_img = resize(img, new_shape, order=3, anti_aliasing=True).astype(np.float32)
    zoomed_label = resize(label, new_shape, order=0, anti_aliasing=False).astype(np.float32)

    if factor < 1:
        pads = [((s - n) // 2, s - n - (s - n) // 2) for s, n in zip(shape, new_shape)]
        img = np.pad(zoomed_img, pads, mode='reflect')
        label = np.pad(zoomed_label, pads, mode='constant', constant_values=2)
    else:
        crop = tuple(slice((n - s) // 2, (n - s) // 2 + s) for s, n in zip(shape, new_shape))
        img = zoomed_img[crop]
        label = zoomed_label[crop]

    return img, label

def contrast_jitter(img, label):
    img *= np.random.uniform(0.8, 1.25)
    return img, label