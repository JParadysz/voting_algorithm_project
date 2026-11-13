from pathlib import Path
from typing import Dict
import numpy as np


class CSVExporter:
  """Eksportuje wyniki symulacji do CSV (nazwy kolumn = klucze słownika)."""

  @staticmethod
  def export(filepath: Path, columns: Dict[str, np.ndarray],
             delimiter: str = ","):
    names = list(columns.keys())
    data = np.column_stack([columns[n] for n in names])
    np.savetxt(
        filepath,
        data,
        delimiter=delimiter,
        header=delimiter.join(names),
        comments="",
        fmt="%.6f",
    )