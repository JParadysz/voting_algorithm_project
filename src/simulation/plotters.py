from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from simulation.engine import SimulationResult


class ResultPlotter:
  """Rysuje wyniki symulacji: odczyty i konsensusy oraz błędy voterów."""

  @staticmethod
  def plot(result: SimulationResult, filepath: Optional[Path] = None,
           show: bool = False):
    fig, (ax_top, ax_err) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Góra: sygnał referencyjny, sensory (cienko) i konsensusy (grubo)
    ax_top.plot(result.time, result.reference, "k--", lw=2,
                label=result.reference_name)
    for sid, row in zip(result.sensor_ids, result.readings):
      ax_top.plot(result.time, row, lw=0.8, alpha=0.6, label=sid)
    for name, c in result.consensus.items():
      ax_top.plot(result.time, c, lw=1.8, label=name)
    ax_top.set_ylabel("Amplituda")
    ax_top.set_title("Odczyty sensorów i wyniki głosowania")
    ax_top.legend(ncol=3, fontsize=8)
    ax_top.grid(alpha=0.3)

    # Dół: błąd każdego votera względem czystego sygnału
    rmse = result.rmse()
    for name, c in result.consensus.items():
      ax_err.plot(result.time, c - result.reference, lw=1.2,
                  label=f"{name} (RMSE={rmse[name]:.3f})")
    ax_err.axhline(0, color="k", lw=0.8)
    ax_err.set_xlabel("Czas [rad]")
    ax_err.set_ylabel("Błąd")
    ax_err.set_title("Błąd względem sygnału referencyjnego")
    ax_err.legend(fontsize=8)
    ax_err.grid(alpha=0.3)

    fig.tight_layout()
    if filepath is not None:
      fig.savefig(filepath, dpi=150)
    if show:
      plt.show()
    plt.close(fig)