from typing import Optional
import numpy as np


class Distortion:
  """Klasa bazowa dla modeli zakłóceń i błędów sensorów."""

  def apply(self, signal: np.ndarray) -> np.ndarray:
    """Modyfikuje sygnał, nakładając zakłócenie."""
    raise NotImplementedError


class GaussianNoise(Distortion):
  """Dodaje szum o rozkładzie normalnym. Parametr seed daje powtarzalność."""

  def __init__(self, mean: float = 0.0, std_dev: float = 0.1,
               seed: Optional[int] = None):
    self.mean = mean
    self.std_dev = std_dev
    self.rng = np.random.default_rng(seed)

  def apply(self, signal: np.ndarray) -> np.ndarray:
    noise = self.rng.normal(self.mean, self.std_dev, size=signal.shape)
    return signal + noise


class ConstantBias(Distortion):
  """Dodaje stałe przesunięcie (bias) do sygnału."""

  def __init__(self, bias_value: float = 0.5):
    self.bias_value = bias_value

  def apply(self, signal: np.ndarray) -> np.ndarray:
    return signal + self.bias_value