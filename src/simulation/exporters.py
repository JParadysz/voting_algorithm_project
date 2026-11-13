import codecs
from pathlib import Path
from typing import Dict
import numpy as np


class CSVExporter:
  """Eksportuje wyniki symulacji do CSV (nazwy kolumn = klucze słownika)."""

  @staticmethod
  def export(filepath: Path, columns: Dict[str, np.ndarray],
             delimiter: str = ",", bom: bool = False):
    """Zapisuje CSV w UTF-8 (polskie znaki w nagłówkach).

    bom=True dopisuje BOM, dzięki czemu Excel na Windowsie poprawnie
    rozpoznaje UTF-8 przy otwarciu pliku dwuklikiem.
    """
    names = list(columns.keys())
    data = np.column_stack([columns[n] for n in names])
    with open(filepath, "wb") as f:
      if bom:
        f.write(codecs.BOM_UTF8)
      np.savetxt(
          f,
          data,
          delimiter=delimiter,
          header=delimiter.join(names),
          comments="",
          fmt="%.6f",
          encoding="utf-8",
      )