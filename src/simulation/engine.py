from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from simulation.generators import SignalGenerator
from simulation.sensors import Sensor
from simulation.voters import Voter


@dataclass
class SimulationResult:
  """Wyniki jednego przebiegu symulacji."""

  time: np.ndarray
  reference: np.ndarray               # czysty sygnał (prawda)
  sensor_ids: List[str]
  readings: np.ndarray                # [num_sensors, num_samples]
  consensus: Dict[str, np.ndarray]    # nazwa votera -> wynik głosowania
  reference_name: str = "Sygnał wzorcowy"  # nazwa sygnału referencyjnego

  def rmse(self) -> Dict[str, float]:
    """Błąd średniokwadratowy każdego votera względem czystego sygnału (ignoruje brak wyniku NaN)."""
    errors: Dict[str, float] = {}
    for name, c in self.consensus.items():
      valid = ~np.isnan(c)
      if np.any(valid):
        errors[name] = float(np.sqrt(np.mean((c[valid] - self.reference[valid]) ** 2)))
      else:
        errors[name] = float("nan")
    return errors

  def to_columns(self) -> Dict[str, np.ndarray]:
    """Kolumny w kolejności: czas, reference, sensory, konsensusy."""
    cols = {"Czas [rad]": self.time, self.reference_name: self.reference}
    for sid, row in zip(self.sensor_ids, self.readings):
      cols[sid] = row
    for name, c in self.consensus.items():
      cols[name] = c
    return cols


class SimulationEngine:
  """Zarządza przebiegiem symulacji: sensory -> głosowanie -> wyniki."""

  def __init__(self, generator: SignalGenerator, sensors: List[Sensor],
               voters: List[Voter]):
    self.generator = generator
    self.sensors = sensors
    self.voters = voters

  def run(self, num_samples: int) -> SimulationResult:
    # Oś czasu i czysty sygnał liczone raz, a następnie współdzielone
    time = self.generator.time_axis(num_samples)
    reference = self.generator.generate(num_samples)

    readings = np.array([s.read(reference) for s in self.sensors])
    consensus = {v.name: v.vote(readings) for v in self.voters}
    return SimulationResult(
        time=time,
        reference=reference,
        sensor_ids=[s.sensor_id for s in self.sensors],
        readings=readings,
        consensus=consensus,
        reference_name=self.generator.name,
    )