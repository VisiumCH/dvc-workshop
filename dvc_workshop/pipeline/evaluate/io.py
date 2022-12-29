import json
import os
from typing import Dict

import tensorflow as tf


def load_model(save_path: str):
    return tf.keras.models.load_model(save_path)


def save_json(data: Dict, save_path: str, file_name: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, file_name), "w") as f:
        json.dump(data, f, indent=2)
