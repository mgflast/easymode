"""Numerical validation of PyTorch vs TensorFlow model equivalence.

This script validates that the PyTorch implementation produces numerically
equivalent results to the TensorFlow implementation.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from easymode.segmentation.model import create as create_tf
from easymode.segmentation.torch_version.model_torch import create as create_torch
from easymode.segmentation.torch_version.torch_utils import tf_to_torch_format, torch_to_tf_format


class NumericalValidator:
    """Validate numerical equivalence between TensorFlow and PyTorch models.

    Parameters
    ----------
    tf_weights_path : str or Path
        Path to TensorFlow .h5 weights file
    torch_weights_path : str or Path
        Path to PyTorch .pth weights file
    device : str, optional
        PyTorch device ('cuda' or 'cpu'), default 'cpu'
    """

    def __init__(self, tf_weights_path, torch_weights_path, device='cpu'):
        self.device = device

        # Load TensorFlow model
        self.tf_model = create_tf()
        dummy_tf = tf.zeros((1, 64, 64, 64, 1))
        _ = self.tf_model(dummy_tf)
        self.tf_model.load_weights(tf_weights_path)

        # Load PyTorch model
        self.torch_model = create_torch()
        try:
            state_dict = torch.load(torch_weights_path, map_location=device)
            missing, unexpected = self.torch_model.load_state_dict(state_dict, strict=False)

            if missing:
                print(f"⚠ Warning: {len(missing)} missing keys in PyTorch state_dict")
                for key in missing[:5]:
                    print(f"  - {key}")
                if len(missing) > 5:
                    print(f"  ... and {len(missing) - 5} more")

            if unexpected:
                print(f"⚠ Warning: {len(unexpected)} unexpected keys in PyTorch state_dict")
                for key in unexpected[:5]:
                    print(f"  - {key}")
                if len(unexpected) > 5:
                    print(f"  ... and {len(unexpected) - 5} more")

        except Exception as e:
            print(f"Error loading PyTorch weights: {e}")
            raise

        self.torch_model.eval()
        self.torch_model = self.torch_model.to(device)

    def validate_weights(self):
        """Validate that weights were loaded correctly.

        Returns
        -------
        bool
            True if validation passed
        """
        print("\n" + "="*60)
        print("WEIGHT VALIDATION")
        print("="*60)

        # Compare parameter counts
        tf_trainable = sum([np.prod(v.shape) for v in self.tf_model.trainable_variables])
        tf_total = sum([np.prod(v.shape) for v in self.tf_model.variables])

        torch_trainable = sum(p.numel() for p in self.torch_model.parameters() if p.requires_grad)
        torch_total = sum(p.numel() for p in self.torch_model.parameters())

        print(f"TensorFlow parameters:")
        print(f"  Trainable: {tf_trainable:,}")
        print(f"  Total: {tf_total:,}")

        print(f"\nPyTorch parameters:")
        print(f"  Trainable: {torch_trainable:,}")
        print(f"  Total: {torch_total:,}")

        if tf_trainable != torch_trainable:
            print("\n✗ FAIL: Trainable parameter count mismatch!")
            return False

        print("\n✓ PASS: Parameter counts match")
        return True

    def validate_forward_pass(self, test_input=None, rtol=1e-4, atol=1e-5):
        """Validate forward pass produces same outputs.

        Parameters
        ----------
        test_input : np.ndarray, optional
            Test input in TF format (D, H, W, 1). If None, random input is used.
        rtol : float, optional
            Relative tolerance for np.allclose
        atol : float, optional
            Absolute tolerance for np.allclose

        Returns
        -------
        bool
            True if validation passed
        """
        print("\n" + "="*60)
        print("FORWARD PASS VALIDATION")
        print("="*60)

        # Create test input
        if test_input is None:
            # Use small random input for speed
            test_input = np.random.randn(64, 128, 128, 1).astype(np.float32)

        print(f"Test input shape (TF format): {test_input.shape}")

        # TensorFlow inference
        tf_input = np.expand_dims(test_input, 0)  # Add batch dim: (1, D, H, W, 1)
        tf_output = self.tf_model(tf_input, training=False).numpy()

        print(f"TensorFlow output shape: {tf_output.shape}")
        print(f"TensorFlow output range: [{tf_output.min():.6f}, {tf_output.max():.6f}]")

        # PyTorch inference
        torch_input = tf_to_torch_format(test_input)  # (1, D, H, W)
        if torch_input.ndim == 4:
            torch_input = torch_input.unsqueeze(0)  # Add batch: (1, 1, D, H, W)
        torch_input = torch_input.to(self.device)

        with torch.no_grad():
            torch_output = self.torch_model(torch_input)

        torch_output_np = torch_to_tf_format(torch_output.cpu())  # Back to TF format

        print(f"PyTorch output shape: {torch_output_np.shape}")
        print(f"PyTorch output range: [{torch_output_np.min():.6f}, {torch_output_np.max():.6f}]")

        # Compare outputs
        abs_diff = np.abs(tf_output - torch_output_np)
        rel_diff = abs_diff / (np.abs(tf_output) + 1e-8)

        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        print(f"\nNumerical Differences:")
        print(f"  Max absolute difference:  {max_abs_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Max relative difference:  {max_rel_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")

        # Check tolerance
        passed = np.allclose(tf_output, torch_output_np, rtol=rtol, atol=atol)

        if passed:
            print(f"\n✓ PASS: Outputs match within tolerance (rtol={rtol}, atol={atol})")
        else:
            print(f"\n✗ FAIL: Outputs exceed tolerance!")

            # Find worst mismatches
            flat_idx = np.argmax(abs_diff)
            idx = np.unravel_index(flat_idx, abs_diff.shape)
            print(f"\n  Worst mismatch at index {idx}:")
            print(f"    TensorFlow: {tf_output[idx]:.8f}")
            print(f"    PyTorch:    {torch_output_np[idx]:.8f}")
            print(f"    Difference: {abs_diff[idx]:.8f}")

            # Show distribution of differences
            percentiles = [50, 90, 95, 99]
            print(f"\n  Absolute difference percentiles:")
            for p in percentiles:
                val = np.percentile(abs_diff, p)
                print(f"    {p}th: {val:.2e}")

        return passed

    def validate_multiple_inputs(self, num_tests=5, rtol=1e-4, atol=1e-5):
        """Validate on multiple random inputs.

        Parameters
        ----------
        num_tests : int, optional
            Number of random inputs to test
        rtol : float, optional
            Relative tolerance
        atol : float, optional
            Absolute tolerance

        Returns
        -------
        bool
            True if all tests passed
        """
        print("\n" + "="*60)
        print(f"MULTIPLE INPUT VALIDATION ({num_tests} tests)")
        print("="*60)

        all_passed = True
        for i in range(num_tests):
            print(f"\nTest {i+1}/{num_tests}:")

            # Different sizes to test
            sizes = [(64, 64, 64, 1), (128, 128, 128, 1)]
            size = sizes[i % len(sizes)]

            test_input = np.random.randn(*size).astype(np.float32)
            passed = self.validate_forward_pass(test_input, rtol=rtol, atol=atol)

            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n✓ ALL TESTS PASSED ({num_tests}/{num_tests})")
        else:
            print(f"\n✗ SOME TESTS FAILED")

        return all_passed


def run_validation(tf_weights, torch_weights, num_tests=5):
    """Main validation entry point.

    Parameters
    ----------
    tf_weights : str or Path
        Path to TensorFlow .h5 weights file
    torch_weights : str or Path
        Path to PyTorch .pth weights file
    num_tests : int, optional
        Number of random input tests to run

    Returns
    -------
    bool
        True if all validations passed
    """
    print("="*60)
    print("PyTorch vs TensorFlow Model Equivalence Validation")
    print("="*60)

    validator = NumericalValidator(tf_weights, torch_weights, device='cpu')

    # 1. Validate weights
    weights_ok = validator.validate_weights()

    # 2. Validate multiple inputs
    inputs_ok = validator.validate_multiple_inputs(num_tests=num_tests)

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Weight validation:  {'✓ PASSED' if weights_ok else '✗ FAILED'}")
    print(f"Forward pass tests: {'✓ PASSED' if inputs_ok else '✗ FAILED'}")

    all_ok = weights_ok and inputs_ok
    if all_ok:
        print("\n🎉 ALL VALIDATIONS PASSED!")
    else:
        print("\n⚠ VALIDATION FAILED - weights may need adjustment")

    return all_ok


def main():
    """Command-line interface for validation."""
    parser = argparse.ArgumentParser(
        description='Validate TensorFlow vs PyTorch model equivalence'
    )
    parser.add_argument(
        '--tf_weights',
        type=str,
        required=True,
        help='TensorFlow .h5 weights file'
    )
    parser.add_argument(
        '--torch_weights',
        type=str,
        required=True,
        help='PyTorch .pth weights file'
    )
    parser.add_argument(
        '--num_tests',
        type=int,
        default=5,
        help='Number of random input tests (default: 5)'
    )

    args = parser.parse_args()

    success = run_validation(
        args.tf_weights,
        args.torch_weights,
        num_tests=args.num_tests
    )

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
