import numpy as np


class SignalGenerator:
  """Klasa bazowa (interfejs) dla generatorów sygnałów."""

  def time_axis(self, num_samples: int) -> np.ndarray:
    """Oś czasu (lub fazy), dla której generowany jest sygnał."""
    raise NotImplementedError

  def generate(self, num_samples: int) -> np.ndarray:
    """Generuje tablicę próbek czystego sygnału."""
    raise NotImplementedError


class SineWaveGenerator(SignalGenerator):
  """Generator fali sinusoidalnej. Domyślnie jeden pełny okres."""

  def __init__(
      self,
      amplitude: float = 1.0,
      periods: float = 1.0,
      phase: float = 0.0,
      offset: float = 0.0,
  ):
    self.amplitude = amplitude
    self.periods = periods
    self.phase = phase
    self.offset = offset

  def time_axis(self, num_samples: int) -> np.ndarray:
    # endpoint=False -> pełny okres bez powtórzenia pierwszej próbki na końcu
    return np.linspace(0, 2 * np.pi * self.periods, num_samples, endpoint=False)

  def generate(self, num_samples: int) -> np.ndarray:
    t = self.time_axis(num_samples)
    return self.offset + self.amplitude * np.sin(t + self.phase)