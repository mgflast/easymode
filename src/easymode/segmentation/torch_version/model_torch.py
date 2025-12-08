"""PyTorch implementation of 3D U-Net segmentation model.

This module provides a PyTorch implementation of the 3D U-Net architecture that is
numerically equivalent to the TensorFlow/Keras implementation in model.py.

Key architectural details:
- Encoder: 6 blocks with filters [32, 64, 128, 256, 512, 1024]
- Decoder: 5 blocks with skip connections
- ResBlock3D: Two 3x3x3 convolutions + BatchNorm + ReLU + residual connection
- BatchNorm parameters match TensorFlow: eps=1e-3, momentum=0.01 (PyTorch definition)
- Input format: NCDHW (channels-first)
- Output: Single channel probability map [0-1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_utils import TFSameConv3d


class ResBlock3D(nn.Module):
    """3D Residual block with batch normalization and ReLU activation.

    This block implements a residual connection with two 3D convolutions,
    each followed by batch normalization. The first conv-BN pair is followed
    by ReLU, and a final ReLU is applied after the residual addition.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First convolution path
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        # BatchNorm parameters matching TensorFlow:
        # TF momentum=0.99 (exponential moving average weight for running stats)
        # PyTorch momentum=0.01 (1 - TF_momentum, weight for current batch)
        # TF epsilon=0.001 (default)
        self.bn1 = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU(inplace=False)  # inplace=False for easier debugging

        # Second convolution path
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)

        # Skip connection adjustment if input/output channels differ
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False
            )
            self.skip_bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.skip_conv = None
            self.skip_bn = None

    def forward(self, x):
        """Forward pass through residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, D, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C_out, D, H, W)
        """
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
            skip = self.skip_bn(skip)
        else:
            skip = x

        # Residual addition and final activation
        out = out + skip
        out = F.relu(out)

        return out


class EncoderBlock(nn.Module):
    """Encoder block with optional downsampling followed by residual block.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int, optional
        Stride for downsampling (default: 1, no downsampling)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride

        # Optional downsampling
        if stride > 1:
            # Use TFSameConv3d to match TensorFlow's asymmetric 'same' padding
            self.downsample = TFSameConv3d(
                in_channels, out_channels,
                kernel_size=3, stride=stride, bias=False
            )
            self.downsample_bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)
            self.downsample_relu = nn.ReLU(inplace=False)
            res_in_channels = out_channels
        else:
            self.downsample = None
            res_in_channels = in_channels

        # Residual block
        self.res_block = ResBlock3D(res_in_channels, out_channels)

    def forward(self, x):
        """Forward pass through encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, D, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C_out, D', H', W')
            where D', H', W' = D/stride, H/stride, W/stride if stride > 1
        """
        if self.downsample is not None:
            x = self.downsample(x)
            x = self.downsample_bn(x)
            x = self.downsample_relu(x)

        x = self.res_block(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with optional upsampling, skip connection, and residual block.

    Parameters
    ----------
    in_channels : int
        Number of input channels from previous decoder layer
    out_channels : int
        Number of output channels
    upsample_kernel_size : int, optional
        Kernel size for upsampling (default: 2)
        If 1, no upsampling is performed
    """

    def __init__(self, in_channels, out_channels, upsample_kernel_size=2):
        super().__init__()
        self.upsample_kernel_size = upsample_kernel_size

        # Optional upsampling
        if upsample_kernel_size > 1:
            self.upsample = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_kernel_size,
                padding=0,  # TF uses padding='same' but with stride=kernel_size, this is equiv to padding=0
                bias=False
            )
            self.upsample_bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)
            self.upsample_relu = nn.ReLU(inplace=False)
            # After concatenating with skip connection, input channels = out_channels * 2
            concat_in_channels = out_channels * 2
        else:
            self.upsample = None
            # After concatenating with skip connection, input channels = in_channels * 2
            concat_in_channels = in_channels * 2

        # Residual block after concatenation
        self.res_block = ResBlock3D(concat_in_channels, out_channels)

    def forward(self, x, skip_connection=None):
        """Forward pass through decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C_in, D, H, W)
        skip_connection : torch.Tensor, optional
            Skip connection from encoder of shape (N, C_skip, D', H', W')

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, C_out, D', H', W')
        """
        # Upsample if needed
        if self.upsample is not None:
            x = self.upsample(x)
            x = self.upsample_bn(x)
            x = self.upsample_relu(x)

        # Concatenate with skip connection
        # PyTorch uses NCDHW format, so concatenate on dim=1 (channel dimension)
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        # Apply residual block
        x = self.res_block(x)
        return x


class UNet(nn.Module):
    """3D U-Net architecture for volumetric segmentation.

    This implements a 3D U-Net with residual blocks, matching the TensorFlow
    implementation. The network consists of:
    - 6 encoder blocks with progressively increasing filters
    - 5 decoder blocks with skip connections
    - Final 1x1 convolution with sigmoid activation

    The architecture uses filters: [32, 64, 128, 256, 512, 1024]
    with downsampling strides: [1, 2, 2, 2, 2, 2]
    and upsampling kernel sizes: [1, 2, 2, 2, 2, 2]

    Input shape: (N, 1, D, H, W) - single channel volumes
    Output shape: (N, 1, D, H, W) - probability maps [0-1]
    """

    def __init__(self):
        super().__init__()

        # Architecture configuration (must match TensorFlow exactly)
        filters = [32, 64, 128, 256, 512, 1024]
        strides = [1, 2, 2, 2, 2, 2]
        upsample_kernel_sizes = [1, 2, 2, 2, 2, 2]

        # Encoder blocks
        self.encoders = nn.ModuleList()
        in_ch = 1  # Single channel input
        for i, (f, s) in enumerate(zip(filters, strides)):
            self.encoders.append(EncoderBlock(in_ch, f, stride=s))
            in_ch = f

        # Decoder blocks
        self.decoders = nn.ModuleList()
        decoder_filters = filters[:-1][::-1]  # [512, 256, 128, 64, 32]
        decoder_upsample = upsample_kernel_sizes[1:][::-1]  # [2, 2, 2, 2, 1]

        in_ch = filters[-1]  # 1024 from bottleneck
        for i, (f, us) in enumerate(zip(decoder_filters, decoder_upsample)):
            self.decoders.append(DecoderBlock(in_ch, f, upsample_kernel_size=us))
            in_ch = f

        # Final output layer
        self.final_conv = nn.Conv3d(decoder_filters[-1], 1, kernel_size=1)

    def forward(self, x):
        """Forward pass through U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 1, D, H, W)

        Returns
        -------
        torch.Tensor
            Output probability map of shape (N, 1, D, H, W)
        """
        # Encoder path - collect outputs for skip connections
        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)

        # Decoder path with skip connections
        # Skip connections are from all encoder outputs except the bottleneck,
        # in reverse order
        skip_connections = encoder_outputs[:-1][::-1]  # Reverse, exclude bottleneck
        x = encoder_outputs[-1]  # Start from bottleneck

        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder(x, skip_connection=skip)

        # Final output with sigmoid activation
        output = torch.sigmoid(self.final_conv(x))

        return output

    def count_parameters(self):
        """Count total and trainable parameters.

        Returns
        -------
        tuple[int, int]
            (total_parameters, trainable_parameters)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create():
    """Create and return a UNet model instance.

    Returns
    -------
    UNet
        Initialized U-Net model

    Notes
    -----
    This function provides a factory method matching the TensorFlow implementation's
    create() function for API compatibility.
    """
    model = UNet()
    return model
