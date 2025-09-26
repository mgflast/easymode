import tensorflow as tf

def fft3d(x):
    x = tf.convert_to_tensor(x)
    if not x.dtype.is_complex:
        x = tf.cast(x, tf.complex64)
    return tf.signal.fftshift(tf.signal.fft3d(x), axes=[-3, -2, -1])

def ifft3d(x):
    x = tf.convert_to_tensor(x)
    return tf.signal.ifft3d(tf.signal.ifftshift(x, axes=[-3, -2, -1]))

def DDWLoss(input_stack, y_pred):
    y_true = input_stack[..., 0]

    l1_loss = tf.reduce_sum(tf.math.abs(y_true - y_pred)) / tf.size(y_true, out_type=tf.float32)
    return l1_loss
    #
    # y_true = input_stack[..., 0]
    # mask = tf.cast(input_stack[..., 1], tf.complex64)
    # mask_rot = tf.cast(input_stack[..., 2], tf.complex64)
    #
    # y_pred = tf.squeeze(y_pred, axis=-1)
    # difference_map = y_true - y_pred
    # fourier_difference = fft3d(tf.cast(difference_map, tf.complex64))
    #
    # denoising_term = mask * mask_rot * fourier_difference
    # dewedging_term = 2 * (1 - mask) * mask_rot * fourier_difference
    #
    # denoising_term = tf.reduce_mean(tf.square(tf.abs(tf.math.real(ifft3d(denoising_term)))))
    # dewedging_term = tf.reduce_mean(tf.square(tf.abs(tf.math.real(ifft3d(dewedging_term)))))
    #
    # return denoising_term + dewedging_term