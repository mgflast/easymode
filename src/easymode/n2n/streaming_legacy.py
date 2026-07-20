"""Bounded-memory helpers for exact legacy denoiser tile assembly.

The upstream denoiser extracts every 160^3 tile into one large NumPy array,
predicts all tiles in chunks, retains every prediction, and only then copies the
central 96^3 region of each prediction into the output volume.  For a full
cryo-ET tomogram, the input and output tile collections can consume tens of
GiB each.

This module preserves the original tile coordinates, zero padding, tile order,
central-core selection, edge truncation, and hard (non-overlapping) placement.
It changes only the memory strategy: a bounded tile buffer is generated,
predicted, and assembled before that buffer is reused.

There is deliberately no weighted overlap-add logic here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Tuple

import numpy as np

Position = Tuple[int, int, int]


def legacy_stride(patch_size: int, overlap: int) -> int:
    """Return the legacy core size/stride after validating the geometry."""

    patch_size = int(patch_size)
    overlap = int(overlap)

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    stride = patch_size - 2 * overlap
    if stride <= 0:
        raise ValueError(
            "invalid legacy tiling geometry: "
            f"patch_size={patch_size}, overlap={overlap}, stride={stride}"
        )
    return stride


def legacy_grid_shape(
    shape: Sequence[int],
    patch_size: int,
    overlap: int,
) -> tuple[int, int, int]:
    """Return the exact (Z, Y, X) tile-grid shape used by upstream Easymode."""

    if len(shape) != 3:
        raise ValueError(f"expected a 3-D shape, received {tuple(shape)!r}")

    dims = tuple(int(value) for value in shape)
    if any(value <= 0 for value in dims):
        raise ValueError(f"all volume dimensions must be positive: {dims!r}")

    stride = legacy_stride(patch_size, overlap)
    return tuple(max(1, (dim + stride - 1) // stride) for dim in dims)  # type: ignore[return-value]


def legacy_tile_count(
    shape: Sequence[int],
    patch_size: int,
    overlap: int,
) -> int:
    """Return the exact number of legacy tiles for a volume shape."""

    grid = legacy_grid_shape(shape, patch_size, overlap)
    return int(grid[0] * grid[1] * grid[2])


def iter_legacy_tile_chunks(
    volume: np.ndarray,
    *,
    patch_size: int,
    overlap: int,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, tuple[Position, ...]]]:
    """Yield exact upstream tiles in bounded, reusable chunks.

    Each yielded array has shape ``(N, patch_size, patch_size, patch_size, 1)``
    where ``1 <= N <= chunk_size``.  The array is a view of a reusable internal
    buffer and is valid only until the generator is resumed.  Callers must
    predict and consume the chunk before requesting the next one.
    """

    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D volume, received shape {volume.shape!r}")

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    patch_size = int(patch_size)
    overlap = int(overlap)
    stride = legacy_stride(patch_size, overlap)
    d, h, w = (int(value) for value in volume.shape)
    z_boxes, y_boxes, x_boxes = legacy_grid_shape(
        volume.shape,
        patch_size,
        overlap,
    )

    tile_buffer = np.zeros(
        (chunk_size, patch_size, patch_size, patch_size, 1),
        dtype=volume.dtype,
    )
    positions: list[Position] = []
    pending = 0

    for z_idx in range(z_boxes):
        for y_idx in range(y_boxes):
            for x_idx in range(x_boxes):
                z_pos = z_idx * stride
                y_pos = y_idx * stride
                x_pos = x_idx * stride

                z_start = z_pos - overlap
                y_start = y_pos - overlap
                x_start = x_pos - overlap

                vol_z_start = max(0, z_start)
                vol_y_start = max(0, y_start)
                vol_x_start = max(0, x_start)

                vol_z_end = min(d, z_start + patch_size)
                vol_y_end = min(h, y_start + patch_size)
                vol_x_end = min(w, x_start + patch_size)

                tile = tile_buffer[pending, ..., 0]
                tile.fill(0)

                tile_z_start = vol_z_start - z_start
                tile_y_start = vol_y_start - y_start
                tile_x_start = vol_x_start - x_start

                extracted = volume[
                    vol_z_start:vol_z_end,
                    vol_y_start:vol_y_end,
                    vol_x_start:vol_x_end,
                ]

                tile[
                    tile_z_start:tile_z_start + extracted.shape[0],
                    tile_y_start:tile_y_start + extracted.shape[1],
                    tile_x_start:tile_x_start + extracted.shape[2],
                ] = extracted

                positions.append((z_pos, y_pos, x_pos))
                pending += 1

                if pending == chunk_size:
                    yield tile_buffer, tuple(positions)
                    pending = 0
                    positions.clear()

    if pending:
        yield tile_buffer[:pending], tuple(positions)


def place_legacy_predictions(
    output_volume: np.ndarray,
    predictions: np.ndarray,
    positions: Sequence[Position],
    *,
    patch_size: int,
    overlap: int,
) -> None:
    """Place predicted tiles with the exact upstream hard-core semantics."""

    output_volume = np.asarray(output_volume)
    if output_volume.ndim != 3:
        raise ValueError(
            f"expected a 3-D output volume, received {output_volume.shape!r}"
        )

    predictions = np.asarray(predictions)
    if predictions.ndim == 5:
        if predictions.shape[-1] != 1:
            raise ValueError(
                "expected one output channel, received prediction shape "
                f"{predictions.shape!r}"
            )
        predictions = predictions[..., 0]
    elif predictions.ndim != 4:
        raise ValueError(
            "expected predictions with shape (N, Z, Y, X[, 1]), received "
            f"{predictions.shape!r}"
        )

    patch_size = int(patch_size)
    overlap = int(overlap)
    stride = legacy_stride(patch_size, overlap)

    if predictions.shape[0] != len(positions):
        raise ValueError(
            f"received {predictions.shape[0]} predictions for "
            f"{len(positions)} positions"
        )
    if predictions.shape[1:] != (patch_size, patch_size, patch_size):
        raise ValueError(
            "prediction spatial shape does not match the configured tile: "
            f"{predictions.shape[1:]!r} versus "
            f"{(patch_size, patch_size, patch_size)!r}"
        )

    d, h, w = (int(value) for value in output_volume.shape)
    core_end = overlap + stride

    for tile, (z_pos, y_pos, x_pos) in zip(predictions, positions):
        center_region = tile[
            overlap:core_end,
            overlap:core_end,
            overlap:core_end,
        ]

        z_end = min(z_pos + stride, d)
        y_end = min(y_pos + stride, h)
        x_end = min(x_pos + stride, w)

        actual_z = z_end - z_pos
        actual_y = y_end - y_pos
        actual_x = x_end - x_pos

        output_volume[
            z_pos:z_end,
            y_pos:y_end,
            x_pos:x_end,
        ] = center_region[:actual_z, :actual_y, :actual_x]
