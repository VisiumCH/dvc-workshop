"""IO module for the evaluate step."""
import json
import os
from typing import Dict

import tensorflow as tf

from dvc_workshop.pipeline.preprocess.utils import create_path


def load_model(save_path: str) -> tf.keras.Model:
    """Load the model."""
    return tf.keras.models.load_model(save_path)


def save_json(data: Dict, save_path: str, file_name: str) -> None:
    """Save a dictomary to a json file."""
    create_path(save_path)
    with open(os.path.join(save_path, file_name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
