import os
import tensorflow as tf
import easymode.core.config as cfg
from datetime import datetime, timezone
import json

MODEL_CACHE_DIR = cfg.settings["MODEL_DIRECTORY"]

def package_checkpoint(title='', checkpoint_directory='', apix=10.0, apix_z=None, arch=None, normalization=None, output_directory=None):
    output_directory = output_directory or MODEL_CACHE_DIR
    os.makedirs(output_directory, exist_ok=True)

    # Find checkpoint files
    checkpoint_files = [f.replace('.index', '') for f in os.listdir(checkpoint_directory) if f.endswith('.index')]
    if not checkpoint_files:
        raise ValueError(f'No checkpoint (*.index) found in {checkpoint_directory}.')
    checkpoint_path = os.path.join(checkpoint_directory, checkpoint_files[-1])

    # Without an explicit arch, determine it from the title.
    # n2n and ddw share the same UNet -- the difference is the (x, y) supervision,
    # not the layers. We tag arch separately so packaged metadata still records which.
    if arch is None:
        if 'n2n' in title or 'ddw' in title or 'iso' in title:
            arch = 'n2n' if 'n2n' in title else ('ddw' if 'ddw' in title else 'iso')
        elif 'tilt' in title:
            arch = 'tilt'
        else:
            arch_file = os.path.join(checkpoint_directory, 'arch.json')
            if os.path.exists(arch_file):
                with open(arch_file) as f:
                    arch = json.load(f).get('arch')

    if arch in ('n2n', 'ddw', 'iso'):
        from easymode.n2n.model import create
        dummy_input = tf.zeros((1, 160, 160, 160, 1))
    elif arch == 'tilt':
        from easymode.tiltfilter.model import create
        dummy_input = [tf.zeros((1, 256, 256, 1)), tf.zeros((1, 256, 256, 1))]
    else:
        from easymode.segmentation.models import get_arch, resolve_arch
        from easymode.segmentation.normalization import NORM_GLOBAL_MAD
        if normalization is None:
            normalization = NORM_GLOBAL_MAD   # input normalization scheme, tagged for segmentation models only
        arch = resolve_arch(arch)
        arch_info = get_arch(arch)
        print(f'Packaging weights as {arch} segmentation model.')
        create = arch_info['module'].create
        dummy_input = tf.zeros((1, *arch_info['input_shape']))

    model = create()
    _ = model(dummy_input)

    model.load_weights(checkpoint_path).expect_partial()
    weights_path = os.path.join(output_directory, f'{title}.h5')
    model.save_weights(weights_path)

    size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f'Saved {weights_path}. File size: {size_mb:.2f} MB')

    metadata = {
        'apix': apix,
        'apix_z': 10.0 if apix_z is None else apix_z,
        'arch': arch,
        'timestamp':  datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    }
    if normalization is not None:
        metadata['normalization'] = normalization

    with open(os.path.join(output_directory, f'{title}.json'), 'w', encoding='utf-8') as j:
        json.dump(metadata, j, indent=2)
