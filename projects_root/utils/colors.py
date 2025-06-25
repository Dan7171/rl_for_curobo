import numpy as np
from dataclasses import dataclass
@dataclass
class npColors:
    red: np.ndarray = np.array([0.5,0,0])
    green: np.ndarray = np.array([0,0.5,0])
    blue: np.ndarray = np.array([0,0,0.5])
    yellow: np.ndarray = np.array([0.5,0.5,0])
    purple: np.ndarray = np.array([0.5,0,0.5])
    orange: np.ndarray = np.array([0.5,0.3,0])
    pink: np.ndarray = np.array([0.5,0.3,0.5])