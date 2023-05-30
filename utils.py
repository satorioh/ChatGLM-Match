import os
import tensorflow as tf


def get_abs_path(path: str) -> str:
    current_path = os.path.abspath(__file__)
    dir_name, file_name = os.path.split(current_path)
    return os.path.join(dir_name, path)


def get_file_list(path, suffix='.pdf') -> []:
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                file_list.append(os.path.join(root, file))
    return file_list


def tf_limit_memory(memory_limit=1024):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("start limit tf memory")
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
