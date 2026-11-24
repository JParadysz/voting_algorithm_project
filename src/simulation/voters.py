from collections import deque
from typing import Callable, Literal, Optional, Sequence, Union
from abc import ABC, abstractmethod
import numpy as np


PredictionMethod = Union[
    Literal["quadratic", "linear", "mean"],
    Callable[[Sequence[float]], float],
]


def find_majority_consensus(
    sample: np.ndarray, agreement_threshold: float
) -> Optional[float]:
  """Szuka największej grupy co najmniej większości sensorów o rozrzucie <= agreement_threshold.

  Zwraca średnią tej grupy albo None, gdy zgody większościowej brak.
  Wspólny krok algorytmów opartych na głosowaniu większościowym.
  """
  num_sensors = len(sample)
  majority_count = num_sensors // 2 + 1
  sorted_vals = np.sort(sample)

  # Szukamy największego przedziału [i, i + k - 1] (k >= majority_count),
  # w którym różnica skrajnych wartości nie przekracza progu zgody.
  for k in range(num_sensors, majority_count - 1, -1):
    spreads = sorted_vals[k - 1 :] - sorted_vals[: num_sensors - k + 1]
    valid_indices = np.where(spreads <= agreement_threshold)[0]
    if len(valid_indices) > 0:
      best_start = int(valid_indices[np.argmin(spreads[valid_indices])])
      return float(np.mean(sorted_vals[best_start : best_start + k]))

  return None


class Voter(ABC):
  """Klasa bazowa dla algorytmów głosowania."""

  name = "Głosowanie"  # nazwa w nagłówku CSV i legendzie wykresu

  @abstractmethod
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
    """Zgoda większościowa w ramach agreement_threshold (None, gdy brak)."""
    return find_majority_consensus(sample, self.agreement_threshold)

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


class SmoothingVoter(Voter):
  """Algorytm wygładzający (Smoothing Voter).

  Rozszerzenie algorytmu większościowego o test akceptacyjny oparty na założeniu,
  że nieciągłość (zbyt duży skok) między kolejnymi wynikami wskazuje na błąd.

  Działanie dla każdej próbki:
  1. Sprawdza zgodność większościową wariantów w ramach `agreement_threshold`.
     Jeśli jest zgoda, wynikiem jest średnia zgodnej grupy.
  2. W razie braku zgody wybiera jako kandydata wartość najbliższą poprzedniemu
     wynikowi algorytmu.
  3. Jeżeli odległość kandydata od poprzedniego wyniku nie przekracza progu
     wygładzania `smoothing_threshold` (β), kandydat jest wynikiem.
  4. W przeciwnym razie algorytm nie zwraca wyniku (`np.nan`). Punktem odniesienia
     pozostaje ostatni poprawny wynik.

  Założenie: pierwszy cykl kończy się pomyślnie i daje punkt odniesienia. Gdy w pierwszych
  próbkach nie ma zgody większościowej, algorytm zwraca `np.nan` do czasu pierwszej zgody.

  Wersja zmodyfikowana (`accumulate_threshold=True`) przyspiesza powrót do poprawnej pracy
  po błędach przejściowych. Dopuszczalne odchylenie od punktu odniesienia jest zadane przez
  skumulowany próg, czyli przedział [-β, +β], który po każdym cyklu bez wyniku rozszerza się
  o β w stronę, w którą odchylił się kandydat (górna granica +β dla odchylenia dodatniego,
  dolna -β dla ujemnego). Po uzyskaniu wyniku przedział wraca do [-β, +β].
  """

  name = "Algorytm wygładzający"

  def __init__(
      self,
      agreement_threshold: float,
      smoothing_threshold: float,
      accumulate_threshold: bool = False,
  ):
    if agreement_threshold < 0:
      raise ValueError("Próg zgody (agreement_threshold) musi być nieujemny.")
    if smoothing_threshold < 0:
      raise ValueError("Próg wygładzania (smoothing_threshold) musi być nieujemny.")

    self.agreement_threshold = float(agreement_threshold)
    self.smoothing_threshold = float(smoothing_threshold)
    self.accumulate_threshold = bool(accumulate_threshold)
    # Nazwa instancji: obie wersje mogą występować w jednej symulacji
    if self.accumulate_threshold:
      self.name = "Algorytm wygładzający zmodyfikowany"

  def vote(self, readings: np.ndarray) -> np.ndarray:
    if readings.ndim != 2 or readings.shape[0] == 0:
      raise ValueError(
          "Odczyty muszą mieć kształt [num_sensors, num_samples] i co najmniej 1 sensor."
      )

    beta = self.smoothing_threshold
    num_samples = readings.shape[1]
    output = np.full(num_samples, np.nan, dtype=float)

    reference: Optional[float] = None  # ostatni poprawny wynik algorytmu
    lower, upper = -beta, beta         # przedział akceptacji odchylenia od reference

    for t in range(num_samples):
      sample = readings[:, t]

      # Krok 1: zgoda większościowa
      consensus = find_majority_consensus(sample, self.agreement_threshold)

      # Krok 2: brak zgody -> kandydat najbliższy poprzedniemu wynikowi + test akceptacyjny
      if consensus is None and reference is not None:
        candidate = float(sample[np.argmin(np.abs(sample - reference))])
        deviation = candidate - reference
        if lower <= deviation <= upper:
          consensus = candidate
        elif self.accumulate_threshold:
          # Cykl bez wyniku: rozszerzamy przedział w stronę odchylenia
          if deviation > 0:
            upper += beta
          else:
            lower -= beta

      if consensus is not None:
        output[t] = consensus
        reference = consensus
        lower, upper = -beta, beta

    return output