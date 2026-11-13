from typing import List, Optional
import numpy as np
from simulation.distortions import Distortion


class Sensor:
  """Reprezentuje pojedynczy sensor nakładający zakłócenia na podany sygnał."""

  def __init__(
      self,
      sensor_id: str,
      distortions: Optional[List[Distortion]] = None,
  ):
    self.sensor_id = sensor_id
    self.distortions = distortions if distortions is not None else []

  def read(self, clean_signal: np.ndarray) -> np.ndarray:
    """Nakłada sekwencyjnie zakłócenia na czysty sygnał (wejście nie jest modyfikowane)."""
    signal = np.array(clean_signal, dtype=float)  # kopia, bez aliasowania wejścia
    for distortion in self.distortions:
      signal = distortion.apply(signal)
    return signal