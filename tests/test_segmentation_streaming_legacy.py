from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (Path(__file__).resolve().parents[1] / 'src' / 'easymode' / 'segmentation' / 'inference.py')


def _load_module():
    # Minimal TensorFlow stubs sufficient for importing inference.py.
    tf = types.ModuleType('tensorflow')

    class _Logger:
        def setLevel(self, *_args, **_kwargs):
            return None

    class _Optimizer:
        @staticmethod
        def set_experimental_options(_options):
            return None

    class _Experimental:
        @staticmethod
        def set_memory_growth(_device, _enabled):
            return None

    class _Config:
        optimizer = _Optimizer()
        experimental = _Experimental()

        @staticmethod
        def list_physical_devices(_kind=None):
            return []

    class ResourceExhaustedError(Exception):
        pass

    tf.get_logger = lambda: _Logger()
    tf.config = _Config()
    tf.errors = types.SimpleNamespace(ResourceExhaustedError=ResourceExhaustedError)
    sys.modules['tensorflow'] = tf

    keras = types.ModuleType('tensorflow.keras')
    keras.mixed_precision = types.SimpleNamespace()
    sys.modules['tensorflow.keras'] = keras

    easymode = types.ModuleType('easymode')
    easymode.__path__ = []
    core = types.ModuleType('easymode.core')
    core.__path__ = []
    distribution = types.ModuleType('easymode.core.distribution')
    distribution.get_model = lambda *_args, **_kwargs: (None, None)
    distribution.load_model = lambda *_args, **_kwargs: None
    sys.modules['easymode'] = easymode
    sys.modules['easymode.core'] = core
    sys.modules['easymode.core.distribution'] = distribution

    spec = importlib.util.spec_from_file_location('seg_stream_test_module', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SEG = _load_module()


class ArrayTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def numpy(self):
        return self._array


class AffineModel:
    def __init__(self, scale=1.0, offset=0.0):
        self.scale = np.float32(scale)
        self.offset = np.float32(offset)
        self.batch_sizes = []
        self.call_inputs = []

    def __call__(self, x, training=False):
        assert training is False
        arr = np.asarray(x)
        self.batch_sizes.append(arr.shape[0])
        self.call_inputs.append(arr.copy())
        return ArrayTensor(arr * self.scale + self.offset)


def legacy_result(volume, model, patch, overlap):
    tiles, positions, shape = SEG.tile_volume(volume, patch, overlap)
    predictions = []
    for tile in tiles:
        predictions.extend(model(tile[None, ...], training=False).numpy())
    return SEG.detile_volume(np.asarray(predictions), positions, shape, patch, overlap)


def test_geometry_matches_legacy_counts():
    shape = (23, 25, 29)
    patch = (16, 20, 24)
    overlap = (3, 4, 5)
    _, _, stride, boxes = SEG._streaming_geometry(shape, patch, overlap)
    assert stride == (10, 12, 14)
    assert boxes == (3, 3, 3)


def test_default_full_tomogram_tile_count():
    _, _, stride, boxes = SEG._streaming_geometry(
        (544, 1280, 1312), (160, 160, 160), (48, 48, 48)
    )
    assert stride == (64, 64, 64)
    assert boxes == (9, 20, 21)
    assert int(np.prod(boxes)) == 3780


def test_each_streamed_tile_matches_legacy_materialization():
    rng = np.random.default_rng(7)
    volume = rng.normal(size=(23, 25, 29)).astype(np.float32)
    patch = (16, 20, 24)
    overlap = (3, 4, 5)
    legacy_tiles, positions, _ = SEG.tile_volume(volume, patch, overlap)
    buffer = np.empty((1, *patch, 1), dtype=np.float32)
    for expected, position in zip(legacy_tiles, positions):
        got = SEG._extract_streaming_tile(volume, buffer, position, patch, overlap)
        assert np.array_equal(got[0], expected)


def test_streaming_identity_reconstructs_irregular_volume_exactly():
    rng = np.random.default_rng(11)
    volume = rng.normal(size=(23, 25, 29)).astype(np.float32)
    model = AffineModel()
    result = SEG._segment_tomogram_instance(
        volume, model, batch_size=99, tile_size=(16, 20, 24), overlap=(3, 4, 5)
    )
    assert result.dtype == np.float32
    assert np.array_equal(result, volume)
    assert set(model.batch_sizes) == {1}


def test_streaming_matches_legacy_for_affine_prediction_exactly():
    rng = np.random.default_rng(13)
    volume = rng.normal(size=(23, 25, 29)).astype(np.float32)
    patch = (16, 20, 24)
    overlap = (3, 4, 5)
    legacy = legacy_result(volume, AffineModel(1.25, -0.125), patch, overlap)
    streaming = SEG._segment_tomogram_instance(
        volume, AffineModel(1.25, -0.125), 1, patch, overlap
    )
    assert np.array_equal(streaming, legacy)


def test_streaming_matches_legacy_at_partial_final_edges():
    volume = np.arange(17 * 21 * 26, dtype=np.float32).reshape(17, 21, 26)
    patch = (14, 18, 20)
    overlap = (2, 3, 4)
    legacy = legacy_result(volume, AffineModel(0.5, 2.0), patch, overlap)
    streaming = SEG._segment_tomogram_instance(
        volume, AffineModel(0.5, 2.0), 1, patch, overlap
    )
    assert np.array_equal(streaming, legacy)
    assert np.all(np.isfinite(streaming))


def test_streaming_preserves_legacy_tile_traversal_order():
    volume = np.zeros((23, 25, 29), dtype=np.float32)
    patch = (16, 20, 24)
    overlap = (3, 4, 5)
    _, positions, _ = SEG.tile_volume(volume, patch, overlap)
    model = AffineModel()
    SEG._segment_tomogram_instance(volume, model, 1, patch, overlap)
    # Encode the expected first voxel of each legacy tile and compare with model inputs.
    buffer = np.empty((1, *patch, 1), dtype=np.float32)
    expected = []
    for position in positions:
        expected.append(
            SEG._extract_streaming_tile(
                volume, buffer, position, patch, overlap
            ).copy()
        )
    assert len(expected) == len(model.call_inputs)
    for a, b in zip(expected, model.call_inputs):
        assert np.array_equal(a, b)


def test_invalid_zero_stride_is_rejected_cleanly():
    with pytest.raises(ValueError, match='stride'):
        SEG._streaming_geometry((128, 128, 128), (96, 96, 96), (48, 48, 48))


def test_prediction_shape_validation():
    with pytest.raises(RuntimeError, match='output shape'):
        SEG._prediction_as_3d(np.zeros((1, 16, 16, 16, 2), np.float32), (16, 16, 16))
    with pytest.raises(RuntimeError, match='spatial shape'):
        SEG._prediction_as_3d(np.zeros((1, 15, 16, 16, 1), np.float32), (16, 16, 16))


def test_no_weighted_overlap_symbols_in_streaming_path():
    text = MODULE_PATH.read_text()
    assert 'weighted overlap' not in text.lower()
    assert 'blend-stride' not in text
    assert 'STREAMING_SEGMENTATION_MODE = "streaming-legacy-3d"' in text
