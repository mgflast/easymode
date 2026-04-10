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
    elif 'ddw' in title:
        from easymode.ddw.model import create
        arch = 'ddw'
    elif 'tilt' in title:
        from easymode.tiltfilter.model import create
        arch = 'tilt'
    else:
        arch_file = os.path.join(checkpoint_directory, 'arch.json')
        if os.path.exists(arch_file):
            with open(arch_file) as f:
                arch = json.load(f).get('arch', 'old')
        else:
            arch = 'old'
        if arch == 'lite':
            print('Packaging weights as lite segmentation model (shallow GroupNorm).')
            from easymode.segmentation.model_lite import create
        elif arch == 'current':
            print('Packaging weights as current segmentation model (GroupNorm).')
            from easymode.segmentation.model_current import create
        else:
            print('Packaging weights as old segmentation model (BatchNorm).')
            from easymode.segmentation.model_old import create
    model = create()
    if 'tilt' in title:
        _ = model([tf.zeros((1, 256, 256, 1)), tf.zeros((1, 256, 256, 1))])
    elif arch == 'lite':
        _ = model(tf.zeros((1, 96, 96, 96, 1)))
    else:
        _ = model(tf.zeros((1, 160, 160, 160, 1)))

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