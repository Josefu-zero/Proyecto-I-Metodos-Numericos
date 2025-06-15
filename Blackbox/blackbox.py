# -*- coding: utf-8 -*-
"""
Python 3
19 / 05 / 2025
@author: z_tjona
"""

import logging
from sys import stdout
from datetime import datetime
import os
import numpy as np
from tensorflow import keras  # mejor compatibilidad

# Configurar logging
import __main__
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(getattr(__main__, '__file__', 'Interactive session'))
logging.info(datetime.now())

def load_model(model_path: str = "Blackbox/blackbox_S.keras") -> keras.Sequential:
    logging.debug(f"Loading model from {model_path}")
    print("Current working directory:", os.getcwd())
    return keras.models.load_model(model_path)

def predict_point(model: keras.Sequential, x1: float, x2: float) -> float:
    if not isinstance(x1, (int, float)):
        raise TypeError("x1 must be a number")
    if not isinstance(x2, (int, float)):
        raise TypeError("x2 must be a number")
    result = model.predict(np.array([[x1, x2]], dtype=np.float32))
    return float(np.round(result).flatten()[0])

def predict_batch(model: keras.Sequential, x1s: list, x2s: list) -> list:
    if not isinstance(x1s, list) or not isinstance(x2s, list):
        raise TypeError("Inputs must be lists")
    array_input = np.array([x1s, x2s], dtype=np.float32).T
    return model.predict(array_input).round().flatten().tolist()
