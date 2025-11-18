import shutil, os
import tensorflow as tf
import easymode.core.config as cfg

MODEL_CACHE_DIR = cfg.settings["MODEL_DIRECTORY"]

def package_checkpoint(title='', checkpoint_directory='', output_directory='', cache=False):
    if 'n2n' in title:
        from easymode.n2n.model import create
    elif 'ddw' in title:
        from easymode.ddw.model import create
    else:
        from easymode.segmentation.model import create
    checkpoint_files = [f.replace('.index', '') for f in os.listdir(checkpoint_directory) if f.endswith('.index')]
    checkpoint_path = os.path.join(checkpoint_directory, checkpoint_files[-1])

    model = create()
    _ = model(tf.zeros((1, 160, 160, 160, 1)))
    model.load_weights(checkpoint_path).expect_partial()

    os.makedirs(output_directory, exist_ok=True)

    model.save_weights(os.path.join(output_directory, f'{title}.h5'))

    size_mb = os.path.getsize(os.path.join(output_directory, f'{title}.h5')) / (1024 * 1024)
    print(f'Saved {os.path.join(output_directory, title + ".h5")}. File size: {size_mb:.2f} MB')
    if cache:
        shutil.copy2(os.path.join(output_directory, f'{title}.h5'), os.path.join(MODEL_CACHE_DIR, f'{title}.h5'))