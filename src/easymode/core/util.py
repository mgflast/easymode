import numpy as np

def _fourier_bin_2d(x, b):
    j, k = x.shape
    x = x[:j//b*b, :k//b*b]

    F = np.fft.fftshift(np.fft.fft2(x.astype(np.float32), norm='ortho'))
    cj, ck = j//2, k//2

    F = F[cj - j//b//2 : cj + (j//b+1)//2, ck - k//b//2 : ck + (k//b+1)//2]

    y = np.fft.ifft2(np.fft.ifftshift(F), norm='ortho').real
    y *= b**2

    return y.astype(np.float32)


def _fourier_bin_3d(x, b):
    i, j, k = x.shape
    x = x[:i // b * b, :j // b * b, :k // b * b]

    F = np.fft.fftshift(np.fft.fftn(x.astype(np.float32), norm='ortho'))
    ci, cj, ck = i // 2, j // 2, k // 2

    F = F[ci - i // b // 2: ci + (i // b + 1) // 2, cj - j // b // 2: cj + (j // b + 1) // 2, ck - k // b // 2: ck + (k // b + 1) // 2]

    y = np.fft.ifftn(np.fft.ifftshift(F), norm='ortho').real
    y *= b ** 3

    return y.astype(np.float32)

def fourier_bin(x, b):
    if b == 1:
        return x

    if len(x.shape) == 2:
        return _fourier_bin_2d(x, b)
    elif len(x.shape) == 3:
        return _fourier_bin_3d(x, b)
    else:
        raise ValueError(f"Array x must be 2 or 3 dimensional, but has shape {x.shape}")

