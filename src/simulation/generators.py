import numpy as np
from abc import ABC, abstractmethod

class SignalGenerator(ABC):
  """Klasa bazowa (interfejs) dla generatorów sygnałów."""

  name = "Sygnał"  # nazwa w nagłówku CSV i legendzie wykresu

  @abstractmethod
  def time_axis(self, num_samples: int) -> np.ndarray:
    """Oś czasu (lub fazy), dla której generowany jest sygnał."""
    raise NotImplementedError

  @abstractmethod
  def generate(self, num_samples: int) -> np.ndarray:
    """Generuje tablicę próbek czystego sygnału."""
    raise NotImplementedError


class SineWaveGenerator(SignalGenerator):
  """Generator fali sinusoidalnej. Domyślnie jeden pełny okres."""

  name = "Sygnał sinusoidalny"

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

class SquareWaveGenerator(SignalGenerator):
  """Generator fali prostokątnej."""

  name = "Sygnał prostokątny"

  def __init__(
      self,
      amplitude: float = 1.0,
      periods: float = 1.0,
      offset: float = 0.0,
      t_max: float = 50.0,
  ):
    self.amplitude = amplitude
    self.periods = periods
    self.offset = offset
    self.t_max = t_max

  def time_axis(self, num_samples: int) -> np.ndarray:
    return np.linspace(0, self.t_max, num_samples)

  def generate(self, num_samples: int) -> np.ndarray:
    t = self.time_axis(num_samples)
    omega = 2 * np.pi * self.periods / self.t_max
    return self.offset + self.amplitude * np.sign(np.sin(omega * t))

class ExponentialResponseGenerator(SignalGenerator):
  """Generator odpowiedzi wykładniczej (np. charakterystyki cieplnej)."""

  name = "Odpowiedź wykładnicza"

  def __init__(
      self,
      T_initial: float = 0.0,
      T_target: float = 100.0,
      tau: float = 5.0,
      delay: float = 10.0,
      t_max: float = 50.0,
  ):
    self.T_initial = T_initial
    self.T_target = T_target
    self.tau = tau
    self.delay = delay
    self.t_max = t_max

  def time_axis(self, num_samples: int) -> np.ndarray:
    return np.linspace(0, self.t_max, num_samples)

  def generate(self, num_samples: int) -> np.ndarray:
    t = self.time_axis(num_samples)
    
    return np.where(
        t < self.delay,
        self.T_initial,
        self.T_target + (self.T_initial - self.T_target) * np.exp(-(t - self.delay) / self.tau)
    )