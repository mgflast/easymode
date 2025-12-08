"""Utility functions for PyTorch segmentation model.

This module provides helper functions for converting between TensorFlow and PyTorch
tensor formats, as well as other utilities specific to the PyTorch implementation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def tf_to_torch_format(volume):
    """Convert TensorFlow NDHWC format to PyTorch NCDHW format.

    Parameters
    ----------
    volume : numpy.ndarray
        Input volume in TensorFlow format:
        - 4D: (D, H, W, C) - single volume with channels
        - 5D: (N, D, H, W, C) - batch of volumes with channels

    Returns
    -------
    torch.Tensor
        Output tensor in PyTorch format:
        - 4D: (C, D, H, W) - single volume with channels
        - 5D: (N, C, D, H, W) - batch of volumes with channels

    Notes
    -----
    TensorFlow uses channels-last format (NDHWC), while PyTorch uses
    channels-first format (NCDHW). This function handles the conversion.
    """
    if volume.ndim == 4:  # (D, H, W, C) -> (C, D, H, W)
        volume = np.transpose(volume, (3, 0, 1, 2))
    elif volume.ndim == 5:  # (N, D, H, W, C) -> (N, C, D, H, W)
        volume = np.transpose(volume, (0, 4, 1, 2, 3))
    elif volume.ndim == 3:  # (D, H, W) -> (1, D, H, W)
        # Single channel volume without explicit channel dimension
        volume = np.expand_dims(volume, axis=0)
    else:
        raise ValueError(
            f"Expected 3D, 4D, or 5D array, got shape {volume.shape}"
        )

    return torch.from_numpy(volume).float()


def torch_to_tf_format(tensor):
    """Convert PyTorch NCDHW format to TensorFlow NDHWC format.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in PyTorch format:
        - 4D: (C, D, H, W) - single volume with channels
        - 5D: (N, C, D, H, W) - batch of volumes with channels

    Returns
    -------
    numpy.ndarray
        Output array in TensorFlow format:
        - 4D: (D, H, W, C) - single volume with channels
        - 5D: (N, D, H, W, C) - batch of volumes with channels

    Notes
    -----
    PyTorch uses channels-first format (NCDHW), while TensorFlow uses
    channels-last format (NDHWC). This function handles the conversion.
    """
    # Convert to numpy first
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor

    if array.ndim == 4:  # (C, D, H, W) -> (D, H, W, C)
        array = np.transpose(array, (1, 2, 3, 0))
    elif array.ndim == 5:  # (N, C, D, H, W) -> (N, D, H, W, C)
        array = np.transpose(array, (0, 2, 3, 4, 1))
    else:
        raise ValueError(
            f"Expected 4D or 5D array, got shape {array.shape}"
        )

    return array


def prepare_model_for_inference(model, device='cuda'):
    """Prepare PyTorch model for inference.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to prepare
    device : str, optional
        Device to move model to ('cuda' or 'cpu'), by default 'cuda'

    Returns
    -------
    torch.nn.Module
        Model in evaluation mode on the specified device

    Notes
    -----
    This function:
    1. Sets model to evaluation mode (important for BatchNorm and Dropout)
    2. Moves model to the specified device
    """
    model.eval()

    # Check if CUDA is available if requested
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        device = 'cpu'

    model = model.to(device)
    return model


def get_device(gpu_id=None):
    """Get PyTorch device for computation.

    Parameters
    ----------
    gpu_id : int, optional
        GPU device ID to use. If None, uses CUDA if available, else CPU.

    Returns
    -------
    torch.device
        PyTorch device object
    """
    if gpu_id is not None:
        if torch.cuda.is_available():
            return torch.device(f'cuda:{gpu_id}')
        else:
            print(f"Warning: GPU {gpu_id} requested but CUDA not available, using CPU")
            return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def clear_gpu_memory():
    """Clear PyTorch GPU memory cache.

    This is useful when running multiple inferences or handling OOM errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_gpu_memory_stats():
    """Print current GPU memory usage statistics.

    Useful for debugging memory issues.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("CUDA not available")


def calculate_tf_same_padding_3d(input_shape, kernel_size, stride):
    """Calculate asymmetric padding for TensorFlow-style 'same' padding.

    TensorFlow's 'same' padding with stride > 1 uses asymmetric padding,
    adding extra padding on the right/bottom when needed. This function
    computes the padding tuple needed for F.pad to replicate this behavior.

    Parameters
    ----------
    input_shape : tuple of int
        Spatial dimensions (D, H, W) of the input tensor
    kernel_size : int or tuple of int
        Kernel size for the convolution. Can be a single int (applied to all dims)
        or a tuple of 3 ints (kD, kH, kW)
    stride : int or tuple of int
        Stride for the convolution. Can be a single int (applied to all dims)
        or a tuple of 3 ints (sD, sH, sW)

    Returns
    -------
    tuple of int
        Padding tuple for F.pad in the format:
        (W_left, W_right, H_left, H_right, D_left, D_right)

    Notes
    -----
    TensorFlow 'same' padding formula:
    - output_size = ceil(input_size / stride)
    - total_padding = max(0, (output_size - 1) * stride + kernel_size - input_size)
    - pad_before = total_padding // 2
    - pad_after = total_padding - pad_before

    F.pad expects padding in reverse order: (last_dim, ..., first_dim)
    """
    # Ensure kernel_size and stride are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    # Calculate padding for each dimension
    paddings = []
    for dim_size, k_size, s_size in zip(input_shape, kernel_size, stride):
        # TensorFlow 'same' padding calculation
        output_size = (dim_size + s_size - 1) // s_size  # Ceiling division
        total_padding = max(0, (output_size - 1) * s_size + k_size - dim_size)
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before
        paddings.append((pad_before, pad_after))

    # F.pad expects (W_left, W_right, H_left, H_right, D_left, D_right)
    # Our paddings are in order (D, H, W), so reverse them
    return (
        paddings[2][0], paddings[2][1],  # W_left, W_right
        paddings[1][0], paddings[1][1],  # H_left, H_right
        paddings[0][0], paddings[0][1],  # D_left, D_right
    )


class TFSameConv3d(nn.Module):
    """Conv3d with TensorFlow-style 'same' padding.

    This module wraps nn.Conv3d to provide padding behavior that matches
    TensorFlow's 'same' padding mode when stride > 1. PyTorch's built-in
    padding uses symmetric padding, while TensorFlow uses asymmetric padding
    that may add extra padding on the right/bottom sides.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int or tuple of int
        Size of the convolving kernel
    stride : int or tuple of int, optional
        Stride of the convolution (default: 1)
    bias : bool, optional
        If True, adds a learnable bias to the output (default: False)

    Notes
    -----
    This is critical for converting TensorFlow models to PyTorch when stride > 1,
    as the different padding strategies can cause significant numerical differences.

    For stride=1, this behaves identically to regular Conv3d with padding='same'.

    Examples
    --------
    >>> conv = TFSameConv3d(32, 64, kernel_size=3, stride=2)
    >>> x = torch.randn(1, 32, 64, 64, 64)
    >>> y = conv(x)
    >>> y.shape
    torch.Size([1, 64, 32, 32, 32])
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        # Create conv with no padding - we'll add it manually
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # No built-in padding
            bias=bias
        )

    def forward(self, x):
        """Forward pass with TF-style 'same' padding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, D, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor after convolution
        """
        # Get spatial dimensions (D, H, W)
        D, H, W = x.shape[2:]

        # Calculate padding based on input shape
        padding = calculate_tf_same_padding_3d(
            (D, H, W), self.kernel_size, self.stride
        )

        # Apply asymmetric padding
        x = F.pad(x, padding)

        # Apply convolution (with padding=0 since we already padded)
        return self.conv(x)
