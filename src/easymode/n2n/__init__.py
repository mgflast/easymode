# easymode.n2n: TensorFlow implementation of the noise2noise-style 3D UNet used as
# the backbone for all easymode denoising/dewedging models.
#
# A single architecture (n2n.model.UNet, l1 + ssim loss) is trained on different
# supervised (x, y) pairings, defined in easymode.training.sampler.MODES:
#   - mode='n2n': classic noise2noise on even/odd half-map pairs.
#   - mode='ddw': distillation of the per-dataset DeepDeWedge teachers
#                 (raw tomogram -> wedge-filled, denoised target).
#
# Inference is method-aware via `easymode denoise --method {n2n, ddw}`.
