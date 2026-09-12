import tensorflow as tf


def configure_tensorflow_memory_growth() -> None:
    """Keep the original TensorFlow GPU memory-growth behavior."""

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("使用デバイス: CPU")
        return

    print("使用デバイス: GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
