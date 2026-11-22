from pathlib import Path
import numpy as np
from simulation.presets import SimulationTemperature, SimulationSinus

def main():
  SimulationTemperature.run(100, Path("out_temperature.csv"), Path("out_temperature.png"))
  SimulationSinus.run(100, Path("out_sin.csv"), Path("out_sin.png"))

if __name__ == "__main__":
  main()