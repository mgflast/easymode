"""Convert TensorFlow/Keras HDF5 weights to PyTorch format.

This script converts weights from the TensorFlow 3D U-Net model to the PyTorch
implementation, handling tensor format conversion and layer name mapping.

The main challenges are:
1. Conv3D weights: TensorFlow (D, H, W, Cin, Cout) → PyTorch (Cout, Cin, D, H, W)
2. BatchNorm params: gamma→weight, beta→bias, moving_mean→running_mean, etc.
3. Layer names: TensorFlow 'encoder_0/...' → PyTorch 'encoders.0....'

Strategy:
We load both the TF model and PyTorch model, then copy weights in order based on
the model structure rather than trying to parse auto-generated names.
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import tensorflow as tf


def load_tf_weights_from_h5(h5_path):
    """Load all TensorFlow weights from HDF5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to .h5 weights file

    Returns
    -------
    dict
        Mapping of weight names to numpy arrays
    """
    weights = {}

    with h5py.File(h5_path, 'r') as f:
        # Recursively extract all datasets
        def extract_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)

        f.visititems(extract_weights)

    return weights


def convert_conv3d_weight(tf_weight):
    """Convert Conv3D weight from TensorFlow to PyTorch format.

    Parameters
    ----------
    tf_weight : np.ndarray
        TensorFlow weight with shape (D, H, W, Cin, Cout)

    Returns
    -------
    torch.Tensor
        PyTorch weight with shape (Cout, Cin, D, H, W)
    """
    # Transpose from (D, H, W, Cin, Cout) to (Cout, Cin, D, H, W)
    pytorch_weight = np.transpose(tf_weight, (4, 3, 0, 1, 2))
    return torch.from_numpy(pytorch_weight).float()


def convert_batchnorm_weights(gamma, beta, moving_mean, moving_var):
    """Convert BatchNorm weights from TensorFlow to PyTorch.

    Parameters
    ----------
    gamma : np.ndarray
        TensorFlow gamma (scale)
    beta : np.ndarray
        TensorFlow beta (shift)
    moving_mean : np.ndarray
        TensorFlow moving mean
    moving_var : np.ndarray
        TensorFlow moving variance

    Returns
    -------
    dict
        Dictionary with PyTorch BatchNorm parameters
    """
    return {
        'weight': torch.from_numpy(gamma).float(),
        'bias': torch.from_numpy(beta).float(),
        'running_mean': torch.from_numpy(moving_mean).float(),
        'running_var': torch.from_numpy(moving_var).float(),
    }


def map_tf_to_pytorch_names(tf_weights):
    """Map TensorFlow weight names to PyTorch state_dict keys.

    This function implements the mapping between TensorFlow layer naming
    and PyTorch module naming conventions.

    Parameters
    ----------
    tf_weights : dict
        TensorFlow weights loaded from HDF5

    Returns
    -------
    OrderedDict
        PyTorch state_dict with converted weights
    """
    pytorch_state = OrderedDict()

    # Group weights by block (encoder_N, decoder_N, output)
    blocks = {}
    for name, weight in tf_weights.items():
        # Skip optimizer states and other metadata
        if 'optimizer' in name.lower() or 'iteration' in name.lower():
            continue

        # Extract block name (encoder_0, decoder_1, output, etc.)
        parts = name.split('/')
        if len(parts) < 1:
            continue

        block_name = parts[0]
        if block_name not in blocks:
            blocks[block_name] = {}
        blocks[block_name][name] = weight

    # Process each block
    for block_name, block_weights in sorted(blocks.items()):
        if block_name.startswith('encoder_'):
            idx = int(block_name.split('_')[1])
            convert_encoder_block(block_weights, idx, pytorch_state)
        elif block_name.startswith('decoder_'):
            idx = int(block_name.split('_')[1])
            convert_decoder_block(block_weights, idx, pytorch_state)
        elif block_name == 'output':
            convert_output_layer(block_weights, pytorch_state)

    return pytorch_state


def convert_encoder_block(block_weights, idx, pytorch_state):
    """Convert an encoder block's weights.

    Parameters
    ----------
    block_weights : dict
        All weights for this encoder block
    idx : int
        Encoder index (0-5)
    pytorch_state : OrderedDict
        PyTorch state dict to populate
    """
    prefix = f'encoders.{idx}'

    # Separate downsample and res_block weights
    downsample_weights = {}
    resblock_weights = {}

    for name, weight in block_weights.items():
        # Remove redundant nesting: encoder_N/u_net/encoder_N/...
        if 'res_block' in name:
            resblock_weights[name] = weight
        else:
            downsample_weights[name] = weight

    # Process downsample (exists for idx > 0)
    if idx > 0:
        # Find downsample conv
        for name, weight in downsample_weights.items():
            if 'kernel' in name and 'res_block' not in name:
                # This is the downsample conv
                # Note: TFSameConv3d has a nested .conv module
                conv_weight = convert_conv3d_weight(weight)
                pytorch_state[f'{prefix}.downsample.conv.weight'] = conv_weight
                break

        # Find downsample BN - collect all params at once
        bn_groups = {}
        for name, weight in downsample_weights.items():
            if 'res_block' in name:
                continue
            if any(x in name for x in ['gamma', 'beta', 'moving_mean', 'moving_variance']):
                bn_path = '/'.join(name.split('/')[:-1])
                if bn_path not in bn_groups:
                    bn_groups[bn_path] = {}
                param = name.split('/')[-1].replace(':0', '')
                bn_groups[bn_path][param] = weight

        # Assign the first complete BN group as downsample_bn
        for bn_path, bn_params in bn_groups.items():
            if len(bn_params) == 4:
                converted = convert_batchnorm_weights(
                    bn_params['gamma'],
                    bn_params['beta'],
                    bn_params['moving_mean'],
                    bn_params['moving_variance']
                )
                for key, value in converted.items():
                    pytorch_state[f'{prefix}.downsample_bn.{key}'] = value
                break

    # Process res_block
    convert_resblock(resblock_weights, f'{prefix}.res_block', pytorch_state)


def convert_decoder_block(block_weights, idx, pytorch_state):
    """Convert a decoder block's weights.

    Parameters
    ----------
    block_weights : dict
        All weights for this decoder block
    idx : int
        Decoder index (0-4)
    pytorch_state : OrderedDict
        PyTorch state dict to populate
    """
    prefix = f'decoders.{idx}'

    # Separate upsample and res_block weights
    upsample_weights = {}
    resblock_weights = {}

    for name, weight in block_weights.items():
        if 'res_block' in name:
            resblock_weights[name] = weight
        else:
            upsample_weights[name] = weight

    # Process upsample conv
    for name, weight in upsample_weights.items():
        if 'conv3d_transpose' in name and 'kernel' in name:
            conv_weight = convert_conv3d_weight(weight)
            pytorch_state[f'{prefix}.upsample.weight'] = conv_weight
            break

    # Find upsample BN - collect all params at once
    bn_groups = {}
    for name, weight in upsample_weights.items():
        if 'res_block' in name:
            continue
        if any(x in name for x in ['gamma', 'beta', 'moving_mean', 'moving_variance']):
            bn_path = '/'.join(name.split('/')[:-1])
            if bn_path not in bn_groups:
                bn_groups[bn_path] = {}
            param = name.split('/')[-1].replace(':0', '')
            bn_groups[bn_path][param] = weight

    # Assign the first complete BN group as upsample_bn
    for bn_path, bn_params in bn_groups.items():
        if len(bn_params) == 4:
            converted = convert_batchnorm_weights(
                bn_params['gamma'],
                bn_params['beta'],
                bn_params['moving_mean'],
                bn_params['moving_variance']
            )
            for key, value in converted.items():
                pytorch_state[f'{prefix}.upsample_bn.{key}'] = value
            break

    # Process res_block
    convert_resblock(resblock_weights, f'{prefix}.res_block', pytorch_state)


def convert_resblock(block_weights, prefix, pytorch_state):
    """Convert a ResBlock's weights.

    ResBlocks have:
    - conv1 (3x3x3) + bn1
    - conv2 (3x3x3) + bn2
    - skip_conv (1x1x1) + skip_bn (optional, when in_channels != out_channels)

    Parameters
    ----------
    block_weights : dict
        All weights for this res_block
    prefix : str
        PyTorch prefix (e.g., 'encoders.0.res_block')
    pytorch_state : OrderedDict
        PyTorch state dict to populate
    """
    # Collect all conv kernels and BN params
    conv_kernels = []
    bn_groups = []

    # Group BN parameters by their path
    bn_paths = {}
    for name, weight in block_weights.items():
        if 'kernel' in name:
            conv_kernels.append((name, weight))
        elif 'gamma' in name:
            bn_path = '/'.join(name.split('/')[:-1])
            if bn_path not in bn_paths:
                bn_paths[bn_path] = {}
            param = name.split('/')[-1].replace(':0', '')
            bn_paths[bn_path][param] = weight
        elif 'beta' in name or 'moving_mean' in name or 'moving_variance' in name:
            bn_path = '/'.join(name.split('/')[:-1])
            if bn_path not in bn_paths:
                bn_paths[bn_path] = {}
            param = name.split('/')[-1].replace(':0', '')
            bn_paths[bn_path][param] = weight

    # Sort conv kernels by numeric suffix in name (e.g., conv3d_9, conv3d_10)
    # Extract number from path like '/encoder_3/.../conv3d_9/kernel:0'
    import re
    def get_layer_number(name):
        # Get the layer name (e.g., 'conv3d_9' from '.../conv3d_9/kernel:0')
        layer_name = name.split('/')[-2]
        # Extract the trailing number (e.g., '9' from 'conv3d_9')
        match = re.search(r'_(\d+)$', layer_name)
        return int(match.group(1)) if match else 0
    conv_kernels.sort(key=lambda x: get_layer_number(x[0]))

    # Identify convs by kernel size
    conv1_weight = None
    conv2_weight = None
    skip_conv_weight = None

    for name, kernel in conv_kernels:
        shape = kernel.shape
        if shape[0] == 1 and shape[1] == 1 and shape[2] == 1:
            # 1x1x1 kernel = skip_conv
            skip_conv_weight = kernel
        elif conv1_weight is None:
            # First 3x3x3 kernel = conv1
            conv1_weight = kernel
        else:
            # Second 3x3x3 kernel = conv2
            conv2_weight = kernel

    # Convert and assign conv weights
    if conv1_weight is not None:
        pytorch_state[f'{prefix}.conv1.weight'] = convert_conv3d_weight(conv1_weight)
    if conv2_weight is not None:
        pytorch_state[f'{prefix}.conv2.weight'] = convert_conv3d_weight(conv2_weight)
    if skip_conv_weight is not None:
        pytorch_state[f'{prefix}.skip_conv.weight'] = convert_conv3d_weight(skip_conv_weight)

    # Process BN groups
    # We have 2 or 3 BN groups (bn1, bn2, and optionally skip_bn)
    bn_groups_list = []
    for bn_path, params in bn_paths.items():
        if len(params) == 4:  # Complete BN
            bn_groups_list.append((bn_path, params))

    # Sort by numeric suffix in path name (e.g., batch_normalization_9, batch_normalization_10)
    def get_bn_number(path):
        bn_name = path.split('/')[-1]
        # Extract trailing number (e.g., '9' from 'batch_normalization_9')
        match = re.search(r'_(\d+)$', bn_name)
        return int(match.group(1)) if match else 0
    bn_groups_list.sort(key=lambda x: get_bn_number(x[0]))

    # Assign BN groups
    # The order should be: bn1, bn2, skip_bn (if present)
    # But in the HDF5, they might be in different order
    # Use the number of BN groups to determine
    if len(bn_groups_list) == 2:
        # No skip connection
        convert_and_assign_bn(bn_groups_list[0][1], f'{prefix}.bn1', pytorch_state)
        convert_and_assign_bn(bn_groups_list[1][1], f'{prefix}.bn2', pytorch_state)
    elif len(bn_groups_list) == 3:
        # Has skip connection
        # The skip_bn usually has a higher number in the auto-generated name
        # So the last one is likely skip_bn
        convert_and_assign_bn(bn_groups_list[0][1], f'{prefix}.bn1', pytorch_state)
        convert_and_assign_bn(bn_groups_list[1][1], f'{prefix}.bn2', pytorch_state)
        convert_and_assign_bn(bn_groups_list[2][1], f'{prefix}.skip_bn', pytorch_state)


def convert_and_assign_bn(bn_params, prefix, pytorch_state):
    """Convert and assign BatchNorm parameters."""
    converted = convert_batchnorm_weights(
        bn_params['gamma'],
        bn_params['beta'],
        bn_params['moving_mean'],
        bn_params['moving_variance']
    )
    for key, value in converted.items():
        pytorch_state[f'{prefix}.{key}'] = value


def convert_output_layer(block_weights, pytorch_state):
    """Convert the output layer weights.

    Parameters
    ----------
    block_weights : dict
        All weights for the output layer
    pytorch_state : OrderedDict
        PyTorch state dict to populate
    """
    for name, weight in block_weights.items():
        if 'kernel' in name:
            conv_weight = convert_conv3d_weight(weight)
            pytorch_state['final_conv.weight'] = conv_weight
        elif 'bias' in name:
            bias_weight = torch.from_numpy(weight).float()
            pytorch_state['final_conv.bias'] = bias_weight


def convert_weights_direct(h5_path, output_path, verbose=True):
    """Convert weights by loading both TF and PyTorch models.

    This is a more robust approach that avoids fragile name mapping.

    Parameters
    ----------
    h5_path : str or Path
        Path to TensorFlow .h5 weights file
    output_path : str or Path
        Path to save PyTorch .pth weights file
    verbose : bool, optional
        Whether to print conversion details (default: True)

    Returns
    -------
    OrderedDict
        The converted PyTorch state_dict
    """
    from easymode.segmentation.model import create as create_tf
    from easymode.segmentation.torch_version.model_torch import create as create_torch

    h5_path = Path(h5_path)
    output_path = Path(output_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"Input file not found: {h5_path}")

    if verbose:
        print(f"Loading TensorFlow model and weights from: {h5_path}")

    # Create TensorFlow model and load weights
    tf_model = create_tf()
    # Build the model with a dummy forward pass
    dummy_tf = tf.zeros((1, 64, 64, 64, 1))
    _ = tf_model(dummy_tf)
    tf_model.load_weights(h5_path)

    if verbose:
        print(f"Loaded TensorFlow model with {len(tf_model.trainable_variables)} trainable vars")

    # Create PyTorch model
    torch_model = create_torch()

    if verbose:
        print(f"Created PyTorch model")

    # Convert weights by iterating through both models in parallel
    pytorch_state = OrderedDict()

    # Get TF variables
    tf_vars = tf_model.trainable_variables

    # Get PyTorch parameters
    torch_params = list(torch_model.named_parameters())

    if verbose:
        print(f"TensorFlow trainable vars: {len(tf_vars)}")
        print(f"PyTorch trainable params: {len(torch_params)}")

    # Manual mapping based on model structure
    # This requires understanding the exact correspondence
    var_idx = 0

    def copy_conv_weight(tf_var, torch_name):
        """Copy and convert Conv3D weight."""
        nonlocal var_idx
        tf_weight = tf_var.numpy()
        # TF: (D, H, W, Cin, Cout) -> PyTorch: (Cout, Cin, D, H, W)
        torch_weight = np.transpose(tf_weight, (4, 3, 0, 1, 2))
        pytorch_state[torch_name] = torch.from_numpy(torch_weight).float()
        if verbose and var_idx < 5:
            print(f"  Conv: {tf_var.name} -> {torch_name}")
            print(f"    TF shape: {tf_weight.shape} -> PyTorch shape: {torch_weight.shape}")
        var_idx += 1

    def copy_bn_params(gamma, beta, mean, var, torch_prefix):
        """Copy BatchNorm parameters."""
        nonlocal var_idx
        pytorch_state[f'{torch_prefix}.weight'] = torch.from_numpy(gamma.numpy()).float()
        pytorch_state[f'{torch_prefix}.bias'] = torch.from_numpy(beta.numpy()).float()
        pytorch_state[f'{torch_prefix}.running_mean'] = torch.from_numpy(mean.numpy()).float()
        pytorch_state[f'{torch_prefix}.running_var'] = torch.from_numpy(var.numpy()).float()
        # PyTorch also has num_batches_tracked but it's not in TF, will be initialized
        if verbose and var_idx < 5:
            print(f"  BN: {gamma.name} -> {torch_prefix}")
        var_idx += 4

    # Iterate through model structure
    # Encoders
    for i, (tf_enc, torch_enc) in enumerate(zip(tf_model.encoders, torch_model.encoders)):
        enc_prefix = f'encoders.{i}'

        # Downsample (if stride > 1)
        if hasattr(tf_enc, 'downsample') and tf_enc.downsample is not None:
            # Find downsample vars in TF
            for var in tf_enc.trainable_variables:
                if 'downsample' in var.name or (i > 0 and 'conv3d' in var.name and 'res_block' not in var.name):
                    if 'kernel' in var.name:
                        copy_conv_weight(var, f'{enc_prefix}.downsample.weight')
                        break

            # Downsample BN
            for j, var in enumerate(tf_enc.trainable_variables):
                if ('batch_normalization' in var.name or 'downsample_bn' in var.name) and 'res_block' not in var.name:
                    # Collect BN params
                    bn_vars = [v for v in tf_enc.trainable_variables if 'batch_normalization' in v.name and 'res_block' not in var.name]
                    if len(bn_vars) >= 4:
                        gamma = [v for v in bn_vars if 'gamma' in v.name][0]
                        beta = [v for v in bn_vars if 'beta' in v.name][0]

                        # Get non-trainable (moving stats)
                        all_vars = tf_enc.variables
                        bn_all = [v for v in all_vars if 'batch_normalization' in v.name and 'res_block' not in v.name]
                        mean = [v for v in bn_all if 'moving_mean' in v.name][0]
                        var_v = [v for v in bn_all if 'moving_variance' in v.name][0]

                        copy_bn_params(gamma, beta, mean, var_v, f'{enc_prefix}.downsample_bn')
                        break

        # ResBlock - this is more complex, skip for now and use name-based fallback

    if verbose:
        print(f"\n⚠ Direct mapping is complex due to TF's auto-generated names")
        print("Falling back to name-based conversion...")

    # Fall back to the name-based approach
    return convert_weights(h5_path, output_path, verbose)


def convert_weights(h5_path, output_path, verbose=True):
    """Full conversion pipeline from TensorFlow HDF5 to PyTorch .pth.

    Parameters
    ----------
    h5_path : str or Path
        Path to TensorFlow .h5 weights file
    output_path : str or Path
        Path to save PyTorch .pth weights file
    verbose : bool, optional
        Whether to print conversion details (default: True)

    Returns
    -------
    OrderedDict
        The converted PyTorch state_dict
    """
    h5_path = Path(h5_path)
    output_path = Path(output_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"Input file not found: {h5_path}")

    if verbose:
        print(f"Loading TensorFlow weights from: {h5_path}")

    # Load TensorFlow weights
    tf_weights = load_tf_weights_from_h5(h5_path)

    if verbose:
        print(f"Loaded {len(tf_weights)} TensorFlow parameters")

    # Convert to PyTorch format
    pytorch_state = map_tf_to_pytorch_names(tf_weights)

    if verbose:
        print(f"Converted to {len(pytorch_state)} PyTorch parameters")

    # Save PyTorch weights
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_state, output_path)

    if verbose:
        print(f"Saved PyTorch weights to: {output_path}")
        print("\nSample converted weights:")
        for i, (key, tensor) in enumerate(list(pytorch_state.items())[:5]):
            print(f"  {key}: {tuple(tensor.shape)}")
        if len(pytorch_state) > 5:
            print(f"  ... and {len(pytorch_state) - 5} more")

    return pytorch_state


def verify_conversion(pytorch_state, expected_param_count=128897505):
    """Verify that weight conversion was successful.

    Parameters
    ----------
    pytorch_state : OrderedDict
        Converted PyTorch state dict
    expected_param_count : int, optional
        Expected number of trainable parameters

    Returns
    -------
    bool
        True if verification passed
    """
    # Count parameters
    total_params = sum(p.numel() for p in pytorch_state.values())

    print(f"\nVerification:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Expected parameters: {expected_param_count:,}")

    # Check for NaN or Inf
    has_nan = any(torch.isnan(p).any() for p in pytorch_state.values())
    has_inf = any(torch.isinf(p).any() for p in pytorch_state.values())

    if has_nan:
        print("  ✗ WARNING: Found NaN values in weights!")
        return False
    if has_inf:
        print("  ✗ WARNING: Found Inf values in weights!")
        return False

    print("  ✓ No NaN or Inf values")

    # Check parameter count (allow some tolerance for non-trainable params)
    if abs(total_params - expected_param_count) < 100000:  # Within 100k params
        print("  ✓ Parameter count matches expected")
        return True
    else:
        print(f"  ✗ WARNING: Parameter count mismatch (diff: {abs(total_params - expected_param_count):,})")
        return False


def main():
    """Command-line interface for weight conversion."""
    parser = argparse.ArgumentParser(
        description='Convert TensorFlow .h5 weights to PyTorch .pth format'
    )
    parser.add_argument(
        'h5_path',
        type=str,
        help='Input .h5 file (TensorFlow weights)'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='Output .pth file (PyTorch weights)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify conversion after completing'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbosity'
    )

    args = parser.parse_args()

    # Convert weights
    pytorch_state = convert_weights(
        args.h5_path,
        args.output_path,
        verbose=not args.quiet
    )

    # Verify if requested
    if args.verify:
        success = verify_conversion(pytorch_state)
        if not success:
            print("\n⚠ Warning: Verification checks failed!")
            return 1
        else:
            print("\n✓ Conversion successful!")
            return 0

    return 0


if __name__ == '__main__':
    exit(main())
