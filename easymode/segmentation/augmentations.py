import numpy as np
import random
from scipy.ndimage import rotate, gaussian_filter, zoom

from easymode.segmentation.membrain_fourier_augmentations.fourier_augmentations import MissingWedgeMaskAndFourierAmplitudeMatchingCombined

ROT_XZ_YZ_MAX_ANGLE = 10.0
ROT_XY_MAX_ANGLE = 10.0

def rotate_90_xy(img, label):
    k = random.randint(0, 3)
    img = np.rot90(img, k=k, axes=(1, 2))
    label = np.rot90(label, k=k, axes=(1, 2))
    return img, label

def rotate_90_xz(img, label):
    k = random.randint(0, 1) * 2
    img = np.rot90(img, k=k, axes=(0, 2))
    label = np.rot90(label, k=k, axes=(0, 2))
    return img, label

def flip(img, label):
    k = random.choice([None, 0, 1, 2])
    if k is not None:
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
    img = gaussian_filter(img, sigma=random.uniform(0.3, 1.5))
    return img, label


def scale(img, label):
    factor = np.random.uniform(0.9, 1.1)
    box_size = img.shape[0]
    if factor < 1:
        zoomed_img = zoom(img, factor, order=1)
        zoomed_label = zoom(label, factor, order=0)

        pad_width = (box_size - zoomed_img.shape[0]) // 2
        remainder = box_size - zoomed_img.shape[0] - 2 * pad_width

        img = np.pad(zoomed_img, [(pad_width, pad_width + remainder)] * 3, mode='reflect')
        label = np.pad(zoomed_label, [(pad_width, pad_width + remainder)] * 3, mode='reflect')
    else:
        zoomed_img = zoom(img, factor, order=1)
        zoomed_label = zoom(label, factor, order=0)

        center = zoomed_img.shape[0] // 2
        start = center - 80
        end = center + 80

        img = zoomed_img[start:end, start:end, start:end]
        label = zoomed_label[start:end, start:end, start:end]
    return img, label