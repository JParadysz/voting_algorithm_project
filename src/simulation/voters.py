from collections import deque
from typing import Callable, Literal, Optional, Sequence, Union
import numpy as np


PredictionMethod = Union[
    Literal["quadratic", "linear", "mean"],
    Callable[[Sequence[float]], float],
]


class Voter:
  """Klasa bazowa dla algorytmów głosowania."""

  name = "Głosowanie"  # nazwa w nagłówku CSV i legendzie wykresu

  def vote(self, readings: np.ndarray) -> np.ndarray:
    """Agreguje odczyty (kształt: [num_sensors, num_samples])."""
    raise NotImplementedError


class WeightedVoter(Voter):
  """Średnia ważona. Wagi są normalizowane do sumy 1."""

  name = "Algorytm średniej ważonej"

  def __init__(self, weights: Sequence[float]):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or np.any(w < 0) or w.sum() <= 0:
      raise ValueError("Wagi muszą być nieujemne i mieć dodatnią sumę.")
    self.weights = w / w.sum()

  def vote(self, readings: np.ndarray) -> np.ndarray:
    if readings.shape[0] != len(self.weights):
      raise ValueError(
          f"Liczba wag ({len(self.weights)}) != liczba sensorów ({readings.shape[0]})."
      )
    return np.average(readings, axis=0, weights=self.weights)


class PredictorCorrectorVoter(Voter):
  """Algorytm głosowania typu predyktor-korektor (Predictor-Corrector Voter).

  Działanie dla każdej próbki:
  1. Sprawdza zgodność większościową wariantów w ramach `agreement_threshold`.
     Jeśli co najmniej większość sensorów zgadza się ze sobą, wynikiem jest ich średnia.
  2. W razie braku zgody, na podstawie 3 ostatnich wyników głosowania oblicza
     wartość spodziewaną (predykcję).
  3. Jeżeli któreś z wartości zwróconych przez warianty mieszczą się w zakresie
     `prediction_threshold` od wartości spodziewanej, jako wynik głosowania
     wybierana jest wartość najbliższa predykcji (korekcja).
  4. Jeżeli żadna z wartości nie mieści się w progu predykcji (lub brak jeszcze
     3 historycznych wyników), algorytm nie zwraca wyniku (`np.nan`).
  """

  name = "Predyktor pierwszego rzędu"
  HISTORY_SIZE = 3

  def __init__(
      self,
      agreement_threshold: float,
      prediction_threshold: float,
      prediction_method: PredictionMethod = "quadratic",
  ):
    if agreement_threshold < 0:
      raise ValueError("Próg zgody (agreement_threshold) musi być nieujemny.")
    if prediction_threshold < 0:
      raise ValueError("Próg predykcji (prediction_threshold) musi być nieujemny.")

    self.agreement_threshold = float(agreement_threshold)
    self.prediction_threshold = float(prediction_threshold)
    self.prediction_method = prediction_method

  def _find_majority_consensus(self, sample: np.ndarray) -> Optional[float]:
    """Szuka największej grupy co najmniej większości sensorów o rozrzucie <= agreement_threshold."""
    num_sensors = len(sample)
    majority_count = num_sensors // 2 + 1
    sorted_vals = np.sort(sample)

    # Szukamy największego przedziału [i, i + k - 1] (k >= majority_count),
    # w którym różnica skrajnych wartości nie przekracza progu zgody.
    for k in range(num_sensors, majority_count - 1, -1):
      spreads = sorted_vals[k - 1 :] - sorted_vals[: num_sensors - k + 1]
      valid_indices = np.where(spreads <= self.agreement_threshold)[0]
      if len(valid_indices) > 0:
        best_start = int(valid_indices[np.argmin(spreads[valid_indices])])
        return float(np.mean(sorted_vals[best_start : best_start + k]))

    return None

  def _predict(self, history: Sequence[float]) -> float:
    """Oblicza wartość spodziewaną na podstawie 3 ostatnich wyników głosowania."""
    y0, y1, y2 = history[0], history[1], history[2]

    if callable(self.prediction_method):
      return float(self.prediction_method(history))
    if self.prediction_method == "quadratic":
      # Ekstrapolacja wielomianowa 2. stopnia (różnicowa) z 3 punktów (t=0, 1, 2 -> t=3)
      return float(y0 - 3.0 * y1 + 3.0 * y2)
    if self.prediction_method == "linear":
      # Regresja liniowa najmniejszych kwadratów po 3 punktach (t=0, 1, 2 -> t=3)
      return float((4.0 * y2 + y1 - 2.0 * y0) / 3.0)
    if self.prediction_method == "mean":
      return float((y0 + y1 + y2) / 3.0)

    raise ValueError(f"Nieznana metoda predykcji: {self.prediction_method}")

  def vote(self, readings: np.ndarray) -> np.ndarray:
    if readings.ndim != 2 or readings.shape[0] == 0:
      raise ValueError(
          "Odczyty muszą mieć kształt [num_sensors, num_samples] i co najmniej 1 sensor."
      )

    num_samples = readings.shape[1]
    output = np.full(num_samples, np.nan, dtype=float)
    history: deque[float] = deque(maxlen=self.HISTORY_SIZE)

    for t in range(num_samples):
      sample = readings[:, t]

      # Krok 1: próba uzyskania zgody większościowej
      consensus = self._find_majority_consensus(sample)

      # Krok 2: brak zgody -> predykcja z 3 ostatnich wyników i korekcja
      if consensus is None and len(history) == self.HISTORY_SIZE:
        expected = self._predict(history)
        diffs = np.abs(sample - expected)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= self.prediction_threshold:
          consensus = float(sample[min_idx])

      if consensus is not None:
        output[t] = consensus
        history.append(consensus)

    return output