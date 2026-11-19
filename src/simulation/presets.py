import functools
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from simulation.distortions import ConstantBias, GaussianNoise, EMISpikes, TurbulenceNoise, LongTermDrift
from simulation.engine import SimulationEngine, SimulationResult
from simulation.exporters import CSVExporter
from simulation.generators import SineWaveGenerator, ExponentialResponseGenerator
from simulation.plotters import ResultPlotter
from simulation.sensors import Sensor
from simulation.voters import PredictorCorrectorVoter, WeightedVoter

class SimulationPreset(ABC):
  @staticmethod
  def save(func):
    @functools.wraps(func)
    def wrapper(num_samples: int, csv_out_path: Path, plot_out_path: Path):
      result = func(num_samples, csv_out_path, plot_out_path)

      csv_out_path.parent.mkdir(parents=True, exist_ok=True)
      CSVExporter.export(csv_out_path, result.to_columns())
      print(f"Zapisano: {csv_out_path.resolve()}")

      plot_out_path.parent.mkdir(parents=True, exist_ok=True)
      ResultPlotter.plot(result, filepath=plot_out_path, show=False)
      print(f"Wykres zapisano do: {plot_out_path.resolve()}")

      return result
    return wrapper

  @abstractmethod
  def run(
      self, num_samples: int, csv_out_path: Path, plot_out_path: Path
  ) -> SimulationResult:
    pass

class SimulationSinus(SimulationPreset):
  @SimulationPreset.save
  @staticmethod
  def run(
      num_samples: int, csv_out_path: Path, plot_out_path: Path
  ) -> SimulationResult:
    # 1. Wspólny generator sygnału (jeden okres sinusa)
    generator = SineWaveGenerator(amplitude=1.0)

    # 2. Sensory o różnych charakterystykach
    noise_a, noise_b, noise_c = 0.05, 0.2, 0.1
    sensors = [
        Sensor("Czujnik A", [GaussianNoise(std_dev=noise_a, seed=1)]),
        Sensor(
            "Czujnik B",
            [GaussianNoise(std_dev=noise_b, seed=2), ConstantBias(0.4)],
        ),
        Sensor("Czujnik C", [GaussianNoise(std_dev=noise_c, seed=3)]),
    ]

    # 3. Algorytmy głosowania
    weights = [1 / noise_a**2, 1 / noise_b**2, 1 / noise_c**2]
    voters = [
        WeightedVoter(weights),
        PredictorCorrectorVoter(
            agreement_threshold=0.15,
            prediction_threshold=0.25,
            prediction_method="quadratic",
        ),
    ]

    # 4. Symulacja
    engine = SimulationEngine(generator, sensors, voters)
    result = engine.run(num_samples)

    print(f"Odczyty: {result.readings.shape}")
    for name, err in result.rmse().items():
      valid_count = int(np.sum(~np.isnan(result.consensus[name])))
      print(f"  RMSE {name:<20}: {err:.4f} (wyniki: {valid_count}/{num_samples})")

    return result


class SimulationTemperature(SimulationPreset):
  @SimulationPreset.save
  @staticmethod
  def run(
      num_samples: int, csv_out_path: Path, plot_out_path: Path
  ) -> SimulationResult:
    generator = ExponentialResponseGenerator()

    # 2. Sensory o różnych charakterystykach
    noise_a, noise_b, noise_c = 0.05, 0.2, 0.1
    sensors = [
        Sensor("Czujnik A", [GaussianNoise(std_dev=noise_a, seed=1), 
                             EMISpikes(0.05, 0.5),
                             TurbulenceNoise(0.2),
                             ConstantBias(0.1)]),
        Sensor(
            "Czujnik B",
            [GaussianNoise(std_dev=noise_b, seed=2),
             EMISpikes(0.05, 0.5),
             TurbulenceNoise(0.2)
            ],
        ),
        Sensor("Czujnik C", [GaussianNoise(std_dev=noise_c, seed=3), 
                             EMISpikes(0.05, 0.5),
                             TurbulenceNoise(0.2),
                             LongTermDrift(0.1)]),
    ]

    # 3. Algorytmy głosowania
    weights = [1 / noise_a**2, 1 / noise_b**2, 1 / noise_c**2]
    voters = [
        WeightedVoter(weights),
        PredictorCorrectorVoter(
            agreement_threshold=0.15,
            prediction_threshold=0.25,
            prediction_method="quadratic",
        ),
    ]

    # 4. Symulacja
    engine = SimulationEngine(generator, sensors, voters)
    result = engine.run(num_samples)

    print(f"Odczyty: {result.readings.shape}")
    for name, err in result.rmse().items():
      valid_count = int(np.sum(~np.isnan(result.consensus[name])))
      print(f"  RMSE {name:<20}: {err:.4f} (wyniki: {valid_count}/{num_samples})")

    return result