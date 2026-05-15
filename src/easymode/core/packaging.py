import os
import tensorflow as tf
import easymode.core.config as cfg
from datetime import datetime, timezone
import json

MODEL_CACHE_DIR = cfg.settings["MODEL_DIRECTORY"]

def package_checkpoint(title='', checkpoint_directory='', apix=10.0):
    # Find checkpoint files
    checkpoint_files = [f.replace('.index', '') for f in os.listdir(checkpoint_directory) if f.endswith('.index')]
    checkpoint_path = os.path.join(checkpoint_directory, checkpoint_files[-1])

    # Determine which model architecture to use
    if 'n2n' in title:
        from easymode.n2n.model import create
        arch = 'n2n'
        dummy_input = tf.zeros((1, 160, 160, 160, 1))
    elif 'ddw' in title:
        from easymode.ddw.model import create
        arch = 'ddw'
        dummy_input = tf.zeros((1, 160, 160, 160, 1))
    elif 'tilt' in title:
        from easymode.tiltfilter.model import create
        arch = 'tilt'
        dummy_input = [tf.zeros((1, 256, 256, 1)), tf.zeros((1, 256, 256, 1))]
    else:
        from easymode.segmentation.models import get_arch, resolve_arch
        arch_file = os.path.join(checkpoint_directory, 'arch.json')
        raw_arch = None
        if os.path.exists(arch_file):
            with open(arch_file) as f:
                raw_arch = json.load(f).get('arch')
        arch = resolve_arch(raw_arch)
        arch_info = get_arch(arch)
        print(f'Packaging weights as {arch} segmentation model.')
        create = arch_info['module'].create
        dummy_input = tf.zeros((1, *arch_info['input_shape']))

    model = create()
    _ = model(dummy_input)

    model.load_weights(checkpoint_path).expect_partial()
    model.save_weights(os.path.join(MODEL_CACHE_DIR, f'{title}.h5'))

    size_mb = os.path.getsize(os.path.join(MODEL_CACHE_DIR, f'{title}.h5')) / (1024 * 1024)
    print(f'Saved {os.path.join(MODEL_CACHE_DIR, title + ".h5")}. File size: {size_mb:.2f} MB')

    metadata = {
        'apix': apix,
        'apix_z': 10.0,
        'arch': arch,
        'timestamp':  datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    }

    with open(os.path.join(MODEL_CACHE_DIR, f'{title}.json'), 'w', encoding='utf-8') as j:
        json.dump(metadata, j, indent=2)
