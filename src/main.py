from pathlib import Path
import numpy as np

from simulation.distortions import ConstantBias, GaussianNoise
from simulation.engine import SimulationEngine
from simulation.exporters import CSVExporter
from simulation.generators import SineWaveGenerator
from simulation.plotters import ResultPlotter
from simulation.sensors import Sensor
from simulation.voters import PredictorCorrectorVoter, WeightedVoter


def main():
  print("Uruchamianie symulacji algorytmów głosowania...")
  num_samples = 100

  # 1. Wspólny generator sygnału (jeden okres sinusa)
  generator = SineWaveGenerator(amplitude=1.0)

  # 2. Sensory o różnych charakterystykach (seed -> powtarzalne wyniki)
  noise_a, noise_b, noise_c = 0.05, 0.2, 0.1
  sensors = [
      Sensor("Sensor_A", generator, [GaussianNoise(std_dev=noise_a, seed=1)]),
      Sensor("Sensor_B", generator,
             [GaussianNoise(std_dev=noise_b, seed=2), ConstantBias(0.4)]),
      Sensor("Sensor_C", generator, [GaussianNoise(std_dev=noise_c, seed=3)]),
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

  # 5. Eksport
  output_path = Path("simulation_results.csv")
  CSVExporter.export(output_path, result.to_columns())
  print(f"Zapisano: {output_path.resolve()}")

  # 6. Wykres
  plot_path = Path("simulation_plot.png")
  ResultPlotter.plot(result, filepath=plot_path, show=False)
  print(f"Wykres zapisano do: {plot_path.resolve()}")


if __name__ == "__main__":
  main()