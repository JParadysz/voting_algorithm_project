from typing import List, Optional
import numpy as np
from simulation.distortions import Distortion
from simulation.generators import SignalGenerator


class Sensor:
  """Reprezentuje pojedynczy sensor odczytujący sygnał z zakłóceniami."""

  def __init__(
      self,
      sensor_id: str,
      generator: SignalGenerator,
      distortions: Optional[List[Distortion]] = None,
  ):
    self.sensor_id = sensor_id
    self.generator = generator
    self.distortions = distortions if distortions is not None else []

  def read(self, num_samples: int) -> np.ndarray:
    """Generuje czysty sygnał i sekwencyjnie nakłada na niego zakłócenia."""
    signal = self.generator.generate(num_samples)
    for distortion in self.distortions:
      signal = distortion.apply(signal)
    return signal