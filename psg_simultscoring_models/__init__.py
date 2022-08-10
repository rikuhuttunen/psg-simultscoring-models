import tensorflow as tf

from . import utime as _utime


custom_objs = {
    'Conv1DBlock': _utime.Conv1DBlock,
    'Encoder': _utime.Encoder,
    'Upsampling1DBlock': _utime.Upsampling1DBlock,
    'Decoder': _utime.Decoder
}

tf.keras.utils.get_custom_objects().update(custom_objs)
