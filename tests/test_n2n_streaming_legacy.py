import numpy as np
import pytest

from easymode.n2n.streaming_legacy import (
    iter_legacy_tile_chunks,
    legacy_grid_shape,
    legacy_stride,
    legacy_tile_count,
    place_legacy_predictions,
)


def _upstream_tile_volume(volume, patch_size, overlap):
    d, h, w = volume.shape
    stride = patch_size - 2 * overlap

    z_boxes = max(1, (d + stride - 1) // stride)
    y_boxes = max(1, (h + stride - 1) // stride)
    x_boxes = max(1, (w + stride - 1) // stride)

    tiles = []
    positions = []

    for z_idx in range(z_boxes):
        for y_idx in range(y_boxes):
            for x_idx in range(x_boxes):
                z_start = z_idx * stride - overlap
                y_start = y_idx * stride - overlap
                x_start = x_idx * stride - overlap

                vol_z_start = max(0, z_start)
                vol_y_start = max(0, y_start)
                vol_x_start = max(0, x_start)

                vol_z_end = min(d, z_start + patch_size)
                vol_y_end = min(h, y_start + patch_size)
                vol_x_end = min(w, x_start + patch_size)

                extracted = volume[
                    vol_z_start:vol_z_end,
                    vol_y_start:vol_y_end,
                    vol_x_start:vol_x_end,
                ]

                tile = np.zeros(
                    (patch_size, patch_size, patch_size),
                    dtype=volume.dtype,
                )

                tile_z_start = vol_z_start - z_start
                tile_y_start = vol_y_start - y_start
                tile_x_start = vol_x_start - x_start

                tile[
                    tile_z_start:tile_z_start + extracted.shape[0],
                    tile_y_start:tile_y_start + extracted.shape[1],
                    tile_x_start:tile_x_start + extracted.shape[2],
                ] = extracted

                tiles.append(tile)
                positions.append(
                    (z_idx * stride, y_idx * stride, x_idx * stride)
                )

    tiles = np.array(tiles)
    tiles = np.expand_dims(tiles, axis=-1)
    return tiles, positions, volume.shape


def _upstream_detile(
    denoised_tiles,
    positions,
    original_shape,
    patch_size,
    overlap,
):
    d, h, w = original_shape
    output_volume = np.zeros((d, h, w), dtype=np.float32)
    stride = patch_size - 2 * overlap

    if denoised_tiles.ndim == 5:
        denoised_tiles = denoised_tiles.squeeze(-1)

    for tile, (z_pos, y_pos, x_pos) in zip(denoised_tiles, positions):
        center_region = tile[
            overlap:overlap + stride,
            overlap:overlap + stride,
            overlap:overlap + stride,
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

    return output_volume


def test_legacy_geometry():
    assert legacy_stride(160, 32) == 96
    assert legacy_grid_shape((534, 1270, 1312), 160, 32) == (6, 14, 14)
    assert legacy_tile_count((534, 1270, 1312), 160, 32) == 1176


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 7])
def test_streamed_tiles_match_upstream_exactly(chunk_size):
    volume = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    patch_size = 6
    overlap = 2

    expected_tiles, expected_positions, _ = _upstream_tile_volume(
        volume,
        patch_size,
        overlap,
    )

    chunks = []
    streamed_positions = []
    for tiles, positions in iter_legacy_tile_chunks(
        volume,
        patch_size=patch_size,
        overlap=overlap,
        chunk_size=chunk_size,
    ):
        # Copy because the generator intentionally reuses its internal buffer.
        chunks.append(tiles.copy())
        streamed_positions.extend(positions)

    actual_tiles = np.concatenate(chunks, axis=0)
    assert np.array_equal(actual_tiles, expected_tiles)
    assert streamed_positions == expected_positions


@pytest.mark.parametrize('chunk_size', [1, 3, 8])
def test_streaming_hard_assembly_matches_upstream_exactly(chunk_size):
    rng = np.random.default_rng(17)
    volume = rng.normal(size=(9, 11, 13)).astype(np.float32)
    patch_size = 8
    overlap = 2

    upstream_tiles, positions, original_shape = _upstream_tile_volume(
        volume,
        patch_size,
        overlap,
    )
    upstream_predictions = upstream_tiles * np.float32(1.25) + np.float32(0.5)
    expected = _upstream_detile(
        upstream_predictions,
        positions,
        original_shape,
        patch_size,
        overlap,
    )

    actual = np.zeros_like(volume, dtype=np.float32)
    for tiles, chunk_positions in iter_legacy_tile_chunks(
        volume,
        patch_size=patch_size,
        overlap=overlap,
        chunk_size=chunk_size,
    ):
        predictions = tiles * np.float32(1.25) + np.float32(0.5)
        place_legacy_predictions(
            actual,
            predictions,
            chunk_positions,
            patch_size=patch_size,
            overlap=overlap,
        )

    assert np.array_equal(actual, expected)


def test_identity_prediction_reconstructs_irregular_volume_exactly():
    rng = np.random.default_rng(23)
    volume = rng.normal(size=(17, 19, 23)).astype(np.float32)
    output = np.zeros_like(volume, dtype=np.float32)

    for tiles, positions in iter_legacy_tile_chunks(
        volume,
        patch_size=10,
        overlap=3,
        chunk_size=4,
    ):
        place_legacy_predictions(
            output,
            tiles,
            positions,
            patch_size=10,
            overlap=3,
        )

    assert np.array_equal(output, volume)


def test_invalid_geometry_and_shapes_are_rejected():
    with pytest.raises(ValueError):
        legacy_stride(8, 4)
    with pytest.raises(ValueError):
        legacy_grid_shape((1, 2), 8, 2)
    with pytest.raises(ValueError):
        list(
            iter_legacy_tile_chunks(
                np.zeros((3, 3, 3), dtype=np.float32),
                patch_size=8,
                overlap=2,
                chunk_size=0,
            )
        )
