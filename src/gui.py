import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
  sys.path.insert(0, SRC_DIR)

from simulation.gui import main

if __name__ == "__main__":
  main()
