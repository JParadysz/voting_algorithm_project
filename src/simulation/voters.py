from typing import Sequence
import numpy as np


class Voter:
  """Klasa bazowa dla algorytmów głosowania."""

  name = "voter"

  def vote(self, readings: np.ndarray) -> np.ndarray:
    """Agreguje odczyty (kształt: [num_sensors, num_samples])."""
    raise NotImplementedError


class WeightedVoter(Voter):
  """Średnia ważona. Wagi są normalizowane do sumy 1."""

  name = "weighted"

  def __init__(self, weights: Sequence[float]):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or np.any(w < 0) or w.sum() <= 0:
      raise ValueError("Wagi muszą być nieujemne i mieć dodatnią sumę.")
    self.weights = w / w.sum()

  def vote(self, readings: np.ndarray) -> np.ndarray:
    if readings.shape[0] != len(self.weights):
      raise ValueError(
          f"Liczba wag ({len(self.weights)}) != liczba sensorów ({readings.shape[0]})."
      )
    return np.average(readings, axis=0, weights=self.weights)