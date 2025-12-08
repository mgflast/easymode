# easymode.n2n: a tensorflow implementation of Noise2Noise (as in cryoCARE).
# purpose is to train two types of N2N networks on the easymode training dataset.
# the first is for denoising mode 'splits', where we do proper N2N training on even/odd volume pairs.
# the second is for denoising mode 'direct', where we train on full subtomograms and their N2N-denoised counterparts.
# using this second mode should allow users to bypass the generation of even/odd frame and volume splits, saving a lot
# of time and storage space - although perhaps at the cost of some denoising performance.

# networks are trained with l1 + ssim loss.

