from pathlib import Path
import numpy as np
from simulation.presets import SimulationTemperature

def main():
  SimulationTemperature.run(100, Path("out_temperature.csv"), Path("out_temperature.png"))

if __name__ == "__main__":
  main()