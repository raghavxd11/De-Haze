import tensorflow as tf

def print_device_info():
    """Prints information about available compute devices (informational only)."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[Device] GPUs available: {[gpu.name for gpu in gpus]}")
    else:
        print("[Device] No GPU detected by TensorFlow; running on CPU.")
