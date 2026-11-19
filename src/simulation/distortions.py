from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

class Distortion(ABC):
  """Klasa bazowa dla modeli zakłóceń i błędów sensorów."""

  @abstractmethod
  def apply(self, signal: np.ndarray) -> np.ndarray:
    """Modyfikuje sygnał, nakładając zakłócenie."""
    raise NotImplementedError


class GaussianNoise(Distortion):
  """Dodaje szum o rozkładzie normalnym. Parametr seed daje powtarzalność."""

  def __init__(self, mean: float = 0.0, std_dev: float = 0.1,
               seed: Optional[int] = None):
    self.mean = mean
    self.std_dev = std_dev
    self.rng = np.random.default_rng(seed)

  def apply(self, signal: np.ndarray) -> np.ndarray:
    noise = self.rng.normal(self.mean, self.std_dev, size=signal.shape)
    return signal + noise

class ConstantBias(Distortion):
  """Dodaje stałe przesunięcie (bias) do sygnału."""

  def __init__(self, bias_value: float = 0.5):
    self.bias_value = bias_value

  def apply(self, signal: np.ndarray) -> np.ndarray:
    return signal + self.bias_value

class TurbulenceNoise(Distortion):
  """Dodaje wolnozmienny szum turbulentny (skumulowany spacer losowy o zerowej średniej)."""

  def __init__(self, scale: float = 0.02, seed: Optional[int] = None):
    self.scale = scale
    self.rng = np.random.default_rng(seed)

  def apply(self, signal: np.ndarray) -> np.ndarray:
    noise = self.rng.normal(0.0, self.scale, size=signal.shape)
    noise = np.cumsum(noise)
    # Wyzerowanie średniej szumu, aby nie przesuwał całego sygnału
    noise -= np.mean(noise)
    return signal + noise


class EMISpikes(Distortion):
  """Dodaje rzadkie, silne impulsy zakłóceń elektromagnetycznych (EMI)."""

  def __init__(self, probability: float = 0.005, amplitude: float = 0.5, seed: Optional[int] = None):
    self.probability = probability 
    self.amplitude = amplitude      
    self.rng = np.random.default_rng(seed)

  def apply(self, signal: np.ndarray) -> np.ndarray:
    noise = np.zeros_like(signal)
    num_spikes = int(len(signal) * self.probability)
    
    if num_spikes > 0:
      # Losowanie indeksów, w których wystąpią piki (bez powtórzeń)
      emi_indices = self.rng.choice(len(signal), size=num_spikes, replace=False)
      # Losowanie znaku piku (- amplituda lub + amplituda)
      spike_values = self.rng.choice([-self.amplitude, self.amplitude], size=num_spikes)
      noise[emi_indices] = spike_values
      
    return signal + noise

class LongTermDrift(Distortion):
  """Symuluje starzenie się czujnika. Błąd rośnie w czasie, 
  powodując 'rozkraczanie się' wskazań niezależnych sensorów."""

  def __init__(self, drift_intensity: float = 0.001, seed: Optional[int] = None):
    self.drift_intensity = drift_intensity
    self.rng = np.random.default_rng(seed)

  def apply(self, signal: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    if t is None:
      t = np.arange(len(signal))
    
    # Amplituda błędu rośnie liniowo wraz z upływem czasu t
    growing_envelope = self.drift_intensity * t
    
    drift = growing_envelope * self.rng.standard_normal(size=signal.shape)
    return signal + drift

class PowerLineInterference(Distortion):
  """Dodaje zakłócenia elektromagnetyczne z sieci elektrycznej (np. 50 Hz)."""

  def __init__(self, amplitude: float = 0.05, frequency: float = 50.0, phase: float = 0.0):
    self.amplitude = amplitude
    self.frequency = frequency
    self.phase = phase

  def apply(self, signal: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    if t is None:
      # Domyślne założenie osi czasu (np. próbkowanie co 1 ms)
      t = np.arange(len(signal)) / 1000.0
      
    hum = self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)
    return signal + hum

class QuantizationNoise(Distortion):
  """Symuluje dyskretną rozdzielczość przetwornika analogowo-cyfrowego (efekt schodkowy)."""

  def __init__(self, resolution: float = 0.1):
    self.resolution = resolution  # wielkość pojedynczego kwantu (kroku)

  def apply(self, signal: np.ndarray) -> np.ndarray:
    # Zaokrąglenie wartości sygnału do najbliższych wielokrotności rozdzielczości ADC
    quantized = np.round(signal / self.resolution) * self.resolution
    return quantized