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
    elif 'ddw' in title:
        from easymode.ddw.model import create
    elif 'tilt' in title:
        from easymode.tiltfilter.model import create
    else:
        # Check checkpoint size to decide lite vs full
        ckpt_size_mb = sum(            os.path.getsize(os.path.join(checkpoint_directory, f))
            for f in os.listdir(checkpoint_directory)
            if f.startswith(os.path.basename(checkpoint_path))
        ) / (1024 * 1024)
        if ckpt_size_mb < 1500:
            print(f'Packaging weights as lite segmentation model.')
            from easymode.segmentation.model_lite import create
        else:
            print(f'Packaging weights as default segmentation model.')
            from easymode.segmentation.model import create
    model = create()
    if 'tilt' in title:
        _ = model([tf.zeros((1, 256, 256, 1)), tf.zeros((1, 256, 256, 1))])
    else:
        _ = model(tf.zeros((1, 160, 160, 160, 1)))

    model.load_weights(checkpoint_path).expect_partial()
    model.save_weights(os.path.join(MODEL_CACHE_DIR, f'{title}.h5'))

    size_mb = os.path.getsize(os.path.join(MODEL_CACHE_DIR, f'{title}.h5')) / (1024 * 1024)
    print(f'Saved {os.path.join(MODEL_CACHE_DIR, title + ".h5")}. File size: {size_mb:.2f} MB')

    metadata = {
        'apix': apix,
        'timestamp':  datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    }

    with open(os.path.join(MODEL_CACHE_DIR, f'{title}.json'), 'w', encoding='utf-8') as j:
        json.dump(metadata, j, indent=2)