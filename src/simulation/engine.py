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

  def rmse(self) -> Dict[str, float]:
    """Błąd średniokwadratowy każdego votera względem czystego sygnału."""
    return {
        name: float(np.sqrt(np.mean((c - self.reference) ** 2)))
        for name, c in self.consensus.items()
    }

  def to_columns(self) -> Dict[str, np.ndarray]:
    """Kolumny w kolejności: czas, reference, sensory, konsensusy."""
    cols = {"time": self.time, "reference": self.reference}
    for sid, row in zip(self.sensor_ids, self.readings):
      cols[sid] = row
    for name, c in self.consensus.items():
      cols[f"vote_{name}"] = c
    return cols


class SimulationEngine:
  """Zarządza przebiegiem symulacji: sensory -> głosowanie -> wyniki."""

  def __init__(self, generator: SignalGenerator, sensors: List[Sensor],
               voters: List[Voter]):
    self.generator = generator
    self.sensors = sensors
    self.voters = voters

  def run(self, num_samples: int) -> SimulationResult:
    readings = np.array([s.read(num_samples) for s in self.sensors])
    consensus = {v.name: v.vote(readings) for v in self.voters}
    return SimulationResult(
        time=self.generator.time_axis(num_samples),
        reference=self.generator.generate(num_samples),
        sensor_ids=[s.sensor_id for s in self.sensors],
        readings=readings,
        consensus=consensus,
    )