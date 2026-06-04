# easymode.training: subtomogram sampling + training data orchestration.
#
# Defines the *flavours* of tomograms we can sample from (raw, even, odd,
# cryocare-denoised, ddw-corrected) and the *modes* / methods that pair them up
# for supervised training (n2n: even -> odd, ddw: raw -> ddw, etc.). One generic
# sampler walks datasets/, extracts matched (x, y) boxes per method, and writes
# them to training/{method}/volumes_{train,val}/{x,y}/.
#
# The `run(...)` entry point wires both stages (sample + fit) together and is
# what the `easymode denoise_train` CLI subcommand invokes.
import os


def run(method, sample_only=False, train_only=False,
        samples_per_dataset=500, box_size=96, workers=None,
        exclude=(), flip_y_for=(),
        epochs=100, batch_size=32, lr_start=1e-3, lr_end=1e-5,
        gpus='0,1,2,3'):
    """Sample (stage A) then fit (stage B) a denoiser for `method`.

    `method` selects the (x, y) flavour pair in sampler.MODES. The two stages
    can be run independently with --sample-only / --train-only. Stage B pins
    CUDA_VISIBLE_DEVICES before importing tensorflow so MirroredStrategy sees
    only the requested GPUs."""
    do_sample = sample_only or not train_only       # default: both stages
    do_train  = train_only  or not sample_only

    if do_sample:
        from easymode.training.sampler import Sampler
        Sampler(mode=method,
                samples_per_dataset=samples_per_dataset,
                box_size=box_size,
                num_workers=workers,
                exclude=exclude,
                flip_y_for=flip_y_for).generate()

    if do_train:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        from easymode.n2n.train import train_n2n
        train_n2n(mode=method,
                  batch_size=batch_size,
                  box_size=box_size,
                  epochs=epochs,
                  lr_start=lr_start,
                  lr_end=lr_end)
