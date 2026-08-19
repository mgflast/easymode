import os as _os
import logging as _logging

# TensorFlow is loud: device banners, oneDNN/AVX notes, missing TensorRT libraries, ptxas fallbacks,
# and a multi-screen protobuf dump every time tf.data fails to shard by file. None of it is
# actionable, and it buries our own output. Set before any `import tensorflow`, so this module has to
# stay import-side-effect-only and free of heavy imports.
#
# 2 = hide INFO and WARNING from the C++ layer; ERROR and FATAL still print, as do Python
# exceptions and anything easymode prints itself. Run with TF_CPP_MIN_LOG_LEVEL=0 to get it all back.
if "TF_CPP_MIN_LOG_LEVEL" not in _os.environ:
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    _os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")
    _logging.getLogger("tensorflow").setLevel(_logging.ERROR)  # the Python-side deprecation notices
