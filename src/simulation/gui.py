from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from simulation.distortions import (
    ConstantBias,
    Distortion,
    EMISpikes,
    GaussianNoise,
    LongTermDrift,
    PowerLineInterference,
    QuantizationNoise,
    TurbulenceNoise,
)
from simulation.engine import SimulationEngine, SimulationResult
from simulation.exporters import CSVExporter
from simulation.generators import (
    ExponentialResponseGenerator,
    SignalGenerator,
    SineWaveGenerator,
    SquareWaveGenerator,
)
from simulation.plotters import ResultPlotter
from simulation.sensors import Sensor
from simulation.voters import (
    PredictorCorrectorVoter,
    SmoothingVoter,
    Voter,
    WeightedVoter,
)


class _OriginalSizeSaveFigure(Figure):
  """Figura zachowująca przy zapisie oryginalne wymiary (10x8 cali, 150 DPI)."""

  def savefig(self, fname, *, dpi=None, **kwargs):
    orig_size = self.get_size_inches().copy()
    orig_dpi = self.get_dpi()
    target_dpi = 150 if dpi is None or dpi == "figure" else dpi
    try:
      self.set_size_inches(10, 8, forward=False)
      self.tight_layout()
      return super().savefig(fname, dpi=target_dpi, **kwargs)
    finally:
      self.set_size_inches(orig_size, forward=False)
      self.set_dpi(orig_dpi)
      self.tight_layout()


@dataclass
class SensorConfig:
  """Konfiguracja pojedynczego sensora i jego zakłóceń dla GUI."""

  sensor_id: str
  manual_weight: float = 1.0
  gaussian_enabled: bool = True
  gaussian_std: float = 0.1
  gaussian_seed: str = ""
  bias_enabled: bool = False
  bias_value: float = 0.5
  emi_enabled: bool = False
  emi_prob: float = 0.05
  emi_amp: float = 0.5
  turbulence_enabled: bool = False
  turbulence_scale: float = 0.02
  drift_enabled: bool = False
  drift_intensity: float = 0.001
  powerline_enabled: bool = False
  powerline_amp: float = 0.05
  powerline_freq: float = 50.0
  quantization_enabled: bool = False
  quantization_res: float = 0.1

  def build_sensor(self) -> Sensor:
    """Tworzy obiekt Sensor na podstawie bieżącej konfiguracji."""
    distortions: List[Distortion] = []
    if self.gaussian_enabled:
      if self.gaussian_std < 0:
        raise ValueError(f"{self.sensor_id}: Gaussian std_dev must be >= 0.")
      seed = int(self.gaussian_seed) if self.gaussian_seed.strip() else None
      distortions.append(GaussianNoise(std_dev=self.gaussian_std, seed=seed))
    if self.bias_enabled:
      distortions.append(ConstantBias(bias_value=self.bias_value))
    if self.emi_enabled:
      if not 0.0 <= self.emi_prob <= 1.0:
        raise ValueError(f"{self.sensor_id}: EMI probability must be in [0, 1].")
      distortions.append(
          EMISpikes(probability=self.emi_prob, amplitude=self.emi_amp)
      )
    if self.turbulence_enabled:
      if self.turbulence_scale < 0:
        raise ValueError(f"{self.sensor_id}: Turbulence scale must be >= 0.")
      distortions.append(TurbulenceNoise(scale=self.turbulence_scale))
    if self.drift_enabled:
      distortions.append(LongTermDrift(drift_intensity=self.drift_intensity))
    if self.powerline_enabled:
      distortions.append(
          PowerLineInterference(
              amplitude=self.powerline_amp, frequency=self.powerline_freq
          )
      )
    if self.quantization_enabled:
      if self.quantization_res <= 0:
        raise ValueError(f"{self.sensor_id}: Quantization resolution must be > 0.")
      distortions.append(QuantizationNoise(resolution=self.quantization_res))

    return Sensor(self.sensor_id, distortions)

  def distortions_summary(self) -> str:
    """Zwięzły opis aktywnych zakłóceń do wyświetlenia w tabeli."""
    parts: List[str] = []
    if self.gaussian_enabled:
      parts.append(f"Gauss(σ={self.gaussian_std:g})")
    if self.bias_enabled:
      parts.append(f"Bias({self.bias_value:g})")
    if self.emi_enabled:
      parts.append(f"EMI({self.emi_prob:g})")
    if self.turbulence_enabled:
      parts.append(f"Turb({self.turbulence_scale:g})")
    if self.drift_enabled:
      parts.append(f"Drift({self.drift_intensity:g})")
    if self.powerline_enabled:
      parts.append(f"50Hz({self.powerline_amp:g})")
    if self.quantization_enabled:
      parts.append(f"ADC({self.quantization_res:g})")
    return ", ".join(parts) if parts else "None"


SIGNAL_SPECS: Dict[str, List[Tuple[str, str, float]]] = {
    "Sine Wave": [
        ("amplitude", "Amplitude", 1.0),
        ("periods", "Periods", 1.0),
        ("phase", "Phase [rad]", 0.0),
        ("offset", "Offset", 0.0),
    ],
    "Square Wave": [
        ("amplitude", "Amplitude", 1.0),
        ("periods", "Periods", 1.0),
        ("offset", "Offset", 0.0),
        ("t_max", "t_max", 50.0),
    ],
    "Exponential Response": [
        ("T_initial", "T_initial", 0.0),
        ("T_target", "T_target", 100.0),
        ("tau", "Tau (τ)", 5.0),
        ("delay", "Delay", 10.0),
        ("t_max", "t_max", 50.0),
    ],
}


def _make_preset_configs() -> Dict[str, Dict[str, Any]]:
  """Zwraca predefiniowane konfiguracje odpowiadające presets.py."""
  return {
      "Sinus (Preset)": {
          "num_samples": 100,
          "signal_type": "Sine Wave",
          "signal_params": {"amplitude": 1.0, "periods": 1.0, "phase": 0.0, "offset": 0.0},
          "sensors": [
              SensorConfig("Czujnik A", manual_weight=400.0, gaussian_std=0.05, gaussian_seed="1"),
              SensorConfig("Czujnik B", manual_weight=25.0, gaussian_std=0.2, gaussian_seed="2"),
              SensorConfig("Czujnik C", manual_weight=100.0, gaussian_std=0.1, gaussian_seed="3"),
          ],
          "auto_weights": True,
          "pc_agreement": 0.15,
          "pc_prediction": 0.25,
          "pc_method": "quadratic",
          "sm_agreement": 0.15,
          "sm_threshold": 0.25,
          "sm_mod_agreement": 0.15,
          "sm_mod_threshold": 0.25,
      },
      "Temperature (Preset)": {
          "num_samples": 100,
          "signal_type": "Exponential Response",
          "signal_params": {
              "T_initial": 0.0,
              "T_target": 100.0,
              "tau": 5.0,
              "delay": 10.0,
              "t_max": 50.0,
          },
          "sensors": [
              SensorConfig("Czujnik A", manual_weight=1.0, gaussian_std=1.0, gaussian_seed="1"),
              SensorConfig("Czujnik B", manual_weight=0.16, gaussian_std=2.5, gaussian_seed="2"),
              SensorConfig("Czujnik C", manual_weight=0.04, gaussian_std=5.0, gaussian_seed="3"),
          ],
          "auto_weights": True,
          "pc_agreement": 1.0,
          "pc_prediction": 5.0,
          "pc_method": "quadratic",
          "sm_agreement": 1.0,
          "sm_threshold": 5.0,
          "sm_mod_agreement": 1.0,
          "sm_mod_threshold": 5.0,
      },
  }


class VotingSimulationApp(ttk.Frame):
  """Prosty interfejs Tkinter dla symulacji algorytmów głosowania."""

  def __init__(self, master: tk.Tk):
    super().__init__(master)
    self.master = master
    self.master.title("Voting Algorithm Simulation")
    self.master.geometry("1360x880")
    self.master.minsize(980, 660)

    self.pack(fill=tk.BOTH, expand=True)

    self.presets = _make_preset_configs()
    self.sensors: List[SensorConfig] = []
    self.selected_sensor_idx: Optional[int] = None
    self._loading_sensor = False
    self.last_result: Optional[SimulationResult] = None

    self._init_variables()
    self._build_layout()
    self._apply_preset("Sinus (Preset)")
    self.run_simulation()

    self.master.bind("<Control-r>", lambda _e: self.run_simulation())
    self.master.bind("<Command-r>", lambda _e: self.run_simulation())

  def _init_variables(self) -> None:
    self.preset_var = tk.StringVar(value="Sinus (Preset)")
    self.num_samples_var = tk.StringVar(value="100")
    self.signal_type_var = tk.StringVar(value="Sine Wave")
    self.signal_param_vars: Dict[str, tk.StringVar] = {}

    # Zmienne edytora wybranego sensora
    self.s_id_var = tk.StringVar()
    self.s_weight_var = tk.StringVar()
    self.s_gauss_on = tk.BooleanVar(value=True)
    self.s_gauss_std = tk.StringVar()
    self.s_gauss_seed = tk.StringVar()
    self.s_bias_on = tk.BooleanVar()
    self.s_bias_val = tk.StringVar()
    self.s_emi_on = tk.BooleanVar()
    self.s_emi_prob = tk.StringVar()
    self.s_emi_amp = tk.StringVar()
    self.s_turb_on = tk.BooleanVar()
    self.s_turb_scale = tk.StringVar()
    self.s_drift_on = tk.BooleanVar()
    self.s_drift_int = tk.StringVar()
    self.s_pwr_on = tk.BooleanVar()
    self.s_pwr_amp = tk.StringVar()
    self.s_pwr_freq = tk.StringVar()
    self.s_quant_on = tk.BooleanVar()
    self.s_quant_res = tk.StringVar()

    # Zmienne algorytmów głosowania
    self.v_weighted_on = tk.BooleanVar(value=True)
    self.v_auto_weights = tk.BooleanVar(value=True)

    self.v_pc_on = tk.BooleanVar(value=True)
    self.v_pc_agree = tk.StringVar(value="0.15")
    self.v_pc_pred = tk.StringVar(value="0.25")
    self.v_pc_method = tk.StringVar(value="quadratic")

    self.v_sm_on = tk.BooleanVar(value=True)
    self.v_sm_agree = tk.StringVar(value="0.15")
    self.v_sm_thresh = tk.StringVar(value="0.25")

    self.v_sm_mod_on = tk.BooleanVar(value=True)
    self.v_sm_mod_agree = tk.StringVar(value="0.15")
    self.v_sm_mod_thresh = tk.StringVar(value="0.25")

    self.status_var = tk.StringVar(value="Ready.")

  def _build_layout(self) -> None:
    paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)

    # Lewy panel (konfiguracja ze scrollbarem)
    sidebar_container = ttk.Frame(paned, width=395)
    sidebar_container.pack_propagate(False)
    paned.add(sidebar_container, weight=0)

    self._sidebar_canvas = tk.Canvas(sidebar_container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(
        sidebar_container, orient=tk.VERTICAL, command=self._sidebar_canvas.yview
    )
    self._sidebar_frame = ttk.Frame(self._sidebar_canvas, padding=10)
    self._sidebar_frame.bind(
        "<Configure>",
        lambda _e: self._sidebar_canvas.configure(
            scrollregion=self._sidebar_canvas.bbox("all")
        ),
    )
    self._sidebar_window = self._sidebar_canvas.create_window(
        (0, 0), window=self._sidebar_frame, anchor="nw"
    )
    self._sidebar_canvas.bind(
        "<Configure>",
        lambda e: self._sidebar_canvas.itemconfigure(
            self._sidebar_window, width=e.width
        ),
    )
    self._sidebar_canvas.configure(yscrollcommand=scrollbar.set)
    self._sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self._sidebar_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    self._build_preset_section(self._sidebar_frame)
    self._build_signal_section(self._sidebar_frame)
    self._build_sensors_section(self._sidebar_frame)
    self._build_voters_section(self._sidebar_frame)
    self._build_actions_section(self._sidebar_frame)
    self._build_metrics_section(self._sidebar_frame)

    # Prawy panel (wykres + pasek stanu)
    right_panel = ttk.Frame(paned, padding=8)
    paned.add(right_panel, weight=1)
    self._build_results_pane(right_panel)

  def _on_mousewheel(self, event: tk.Event) -> None:
    if event.delta:
      step = -1 if event.delta > 0 else 1
      self._sidebar_canvas.yview_scroll(step, "units")

  def _build_preset_section(self, parent: ttk.Frame) -> None:
    frame = ttk.LabelFrame(parent, text="Preset", padding=8)
    frame.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(frame, text="Configuration:").pack(side=tk.LEFT)
    combo = ttk.Combobox(
        frame,
        textvariable=self.preset_var,
        values=list(self.presets.keys()) + ["Custom"],
        state="readonly",
        width=22,
    )
    combo.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)
    combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

  def _build_signal_section(self, parent: ttk.Frame) -> None:
    frame = ttk.LabelFrame(parent, text="Signal & Simulation", padding=8)
    frame.pack(fill=tk.X, pady=(0, 8))
    frame.columnconfigure(1, weight=1)

    ttk.Label(frame, text="Samples:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(frame, textvariable=self.num_samples_var, width=12).grid(
        row=0, column=1, sticky="ew", padx=(8, 0), pady=2
    )

    ttk.Label(frame, text="Signal Type:").grid(row=1, column=0, sticky="w", pady=2)
    sig_combo = ttk.Combobox(
        frame,
        textvariable=self.signal_type_var,
        values=list(SIGNAL_SPECS.keys()),
        state="readonly",
    )
    sig_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=2)
    sig_combo.bind("<<ComboboxSelected>>", lambda _e: self._rebuild_signal_params())

    self.signal_params_frame = ttk.Frame(frame)
    self.signal_params_frame.grid(
        row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0)
    )
    self.signal_params_frame.columnconfigure(1, weight=1)
    self.signal_params_frame.columnconfigure(3, weight=1)

  def _rebuild_signal_params(
      self, overrides: Optional[Dict[str, float]] = None
  ) -> None:
    for widget in self.signal_params_frame.winfo_children():
      widget.destroy()

    specs = SIGNAL_SPECS[self.signal_type_var.get()]
    existing = {k: v.get() for k, v in self.signal_param_vars.items()}
    self.signal_param_vars.clear()

    for idx, (key, label, default) in enumerate(specs):
      val = overrides[key] if overrides and key in overrides else existing.get(key, str(default))
      var = tk.StringVar(value=str(val))
      self.signal_param_vars[key] = var
      row, col = divmod(idx, 2)
      ttk.Label(self.signal_params_frame, text=f"{label}:").grid(
          row=row, column=col * 2, sticky="w", padx=(0 if col == 0 else 8, 4), pady=2
      )
      ttk.Entry(self.signal_params_frame, textvariable=var, width=9).grid(
          row=row, column=col * 2 + 1, sticky="ew", pady=2
      )

  def _build_sensors_section(self, parent: ttk.Frame) -> None:
    frame = ttk.LabelFrame(parent, text="Sensors & Distortions", padding=8)
    frame.pack(fill=tk.X, pady=(0, 8))

    tree_frame = ttk.Frame(frame)
    tree_frame.pack(fill=tk.X)

    self.sensor_tree = ttk.Treeview(
        tree_frame,
        columns=("id", "distortions"),
        show="headings",
        height=4,
        selectmode="browse",
    )
    self.sensor_tree.heading("id", text="Sensor")
    self.sensor_tree.heading("distortions", text="Distortions")
    self.sensor_tree.column("id", width=105, stretch=False)
    self.sensor_tree.column("distortions", width=230)
    self.sensor_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self.sensor_tree.bind("<<TreeviewSelect>>", self._on_sensor_selected)

    btn_col = ttk.Frame(frame)
    btn_col.pack(fill=tk.X, pady=(4, 6))
    ttk.Button(btn_col, text="+ Add Sensor", command=self._add_sensor).pack(
        side=tk.LEFT, padx=(0, 4)
    )
    ttk.Button(btn_col, text="- Remove", command=self._remove_sensor).pack(
        side=tk.LEFT
    )
    ttk.Button(btn_col, text="Apply Edit", command=self._apply_sensor_edit).pack(
        side=tk.RIGHT
    )

    # Edytor wybranego sensora
    ed = ttk.Frame(frame)
    ed.pack(fill=tk.X)
    ed.columnconfigure(1, weight=1)
    ed.columnconfigure(3, weight=1)

    ttk.Label(ed, text="ID:").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Entry(ed, textvariable=self.s_id_var, width=12).grid(
        row=0, column=1, sticky="ew", padx=(4, 8), pady=2
    )
    ttk.Label(ed, text="Weight:").grid(row=0, column=2, sticky="w", pady=2)
    ttk.Entry(ed, textvariable=self.s_weight_var, width=8).grid(
        row=0, column=3, sticky="ew", padx=(4, 0), pady=2
    )

    distortion_rows = [
        (self.s_gauss_on, "Gaussian σ / seed", self.s_gauss_std, self.s_gauss_seed),
        (self.s_bias_on, "Constant Bias", self.s_bias_val, None),
        (self.s_emi_on, "EMI prob / amp", self.s_emi_prob, self.s_emi_amp),
        (self.s_turb_on, "Turbulence scale", self.s_turb_scale, None),
        (self.s_drift_on, "Long-Term Drift", self.s_drift_int, None),
        (self.s_pwr_on, "Power Line amp / Hz", self.s_pwr_amp, self.s_pwr_freq),
        (self.s_quant_on, "Quantization res", self.s_quant_res, None),
    ]
    for r, (chk_var, label, var1, var2) in enumerate(distortion_rows, start=1):
      ttk.Checkbutton(ed, text=label, variable=chk_var).grid(
          row=r, column=0, columnspan=2, sticky="w", pady=1
      )
      ttk.Entry(ed, textvariable=var1, width=8).grid(
          row=r, column=2, sticky="ew", padx=(4, 2), pady=1
      )
      if var2 is not None:
        ttk.Entry(ed, textvariable=var2, width=8).grid(
            row=r, column=3, sticky="ew", padx=(2, 0), pady=1
        )

  def _build_voters_section(self, parent: ttk.Frame) -> None:
    frame = ttk.LabelFrame(parent, text="Voting Algorithms", padding=8)
    frame.pack(fill=tk.X, pady=(0, 8))
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=1)

    # 1. Weighted Voter
    ttk.Checkbutton(
        frame, text="Weighted Average", variable=self.v_weighted_on
    ).grid(row=0, column=0, sticky="w", pady=2)
    ttk.Checkbutton(
        frame, text="Auto weights (1/σ²)", variable=self.v_auto_weights
    ).grid(row=0, column=1, columnspan=2, sticky="w", pady=2)

    # 2. Predictor-Corrector
    ttk.Checkbutton(
        frame, text="Predictor-Corrector", variable=self.v_pc_on
    ).grid(row=1, column=0, sticky="w", pady=(6, 2))
    ttk.Combobox(
        frame,
        textvariable=self.v_pc_method,
        values=["quadratic", "linear", "mean"],
        state="readonly",
        width=10,
    ).grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 2))

    self._add_two_param_row(
        frame, 2, "  agree / pred thr:", self.v_pc_agree, self.v_pc_pred
    )

    # 3. Smoothing Voter
    ttk.Checkbutton(
        frame, text="Smoothing Voter", variable=self.v_sm_on
    ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 2))
    self._add_two_param_row(
        frame, 4, "  agree / smooth thr:", self.v_sm_agree, self.v_sm_thresh
    )

    # 4. Modified Smoothing Voter
    ttk.Checkbutton(
        frame, text="Modified Smoothing", variable=self.v_sm_mod_on
    ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 2))
    self._add_two_param_row(
        frame, 6, "  agree / smooth thr:", self.v_sm_mod_agree, self.v_sm_mod_thresh
    )

  @staticmethod
  def _add_two_param_row(
      parent: ttk.Frame,
      row: int,
      label: str,
      var1: tk.StringVar,
      var2: tk.StringVar,
  ) -> None:
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
    ttk.Entry(parent, textvariable=var1, width=8).grid(
        row=row, column=1, sticky="ew", padx=(4, 2), pady=1
    )
    ttk.Entry(parent, textvariable=var2, width=8).grid(
        row=row, column=2, sticky="ew", padx=(2, 0), pady=1
    )

  def _build_actions_section(self, parent: ttk.Frame) -> None:
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=(4, 6))

    ttk.Button(
        frame, text="Run Simulation", command=self.run_simulation
    ).pack(fill=tk.X, pady=(0, 6))

    export_row = ttk.Frame(frame)
    export_row.pack(fill=tk.X)
    ttk.Button(
        export_row, text="Export CSV...", command=self.export_csv
    ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
    ttk.Button(
        export_row, text="Native Plot", command=self.open_native_plot
    ).pack(side=tk.LEFT, fill=tk.X, expand=True)

  def _build_metrics_section(self, parent: ttk.Frame) -> None:
    metrics_frame = ttk.LabelFrame(parent, text="RMSE Summary", padding=6)
    metrics_frame.pack(fill=tk.X, pady=(4, 4))

    self.rmse_tree = ttk.Treeview(
        metrics_frame,
        columns=("algorithm", "rmse", "valid"),
        show="headings",
        height=4,
    )
    self.rmse_tree.heading("algorithm", text="Algorithm")
    self.rmse_tree.heading("rmse", text="RMSE")
    self.rmse_tree.heading("valid", text="Valid")
    self.rmse_tree.column("algorithm", width=190)
    self.rmse_tree.column("rmse", width=75, anchor="e")
    self.rmse_tree.column("valid", width=75, anchor="center")
    self.rmse_tree.pack(fill=tk.X, expand=True)

  def _build_results_pane(self, parent: ttk.Frame) -> None:
    self.figure = _OriginalSizeSaveFigure(figsize=(10, 8), dpi=100)
    self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
    toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)

    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    status_bar = ttk.Label(
        parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=(6, 3)
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

  def _on_preset_selected(self, _event: tk.Event) -> None:
    name = self.preset_var.get()
    if name in self.presets:
      self._apply_preset(name)
      self.run_simulation()

  def _apply_preset(self, name: str) -> None:
    cfg = self.presets[name]
    self.num_samples_var.set(str(cfg["num_samples"]))
    self.signal_type_var.set(cfg["signal_type"])
    self._rebuild_signal_params(cfg["signal_params"])

    self.sensors = [SensorConfig(**vars(s)) for s in cfg["sensors"]]
    self.selected_sensor_idx = None
    self._refresh_sensor_tree(select_idx=0)

    self.v_weighted_on.set(True)
    self.v_auto_weights.set(cfg["auto_weights"])
    self.v_pc_on.set(True)
    self.v_pc_agree.set(str(cfg["pc_agreement"]))
    self.v_pc_pred.set(str(cfg["pc_prediction"]))
    self.v_pc_method.set(cfg["pc_method"])
    self.v_sm_on.set(True)
    self.v_sm_agree.set(str(cfg["sm_agreement"]))
    self.v_sm_thresh.set(str(cfg["sm_threshold"]))
    self.v_sm_mod_on.set(True)
    self.v_sm_mod_agree.set(str(cfg["sm_mod_agreement"]))
    self.v_sm_mod_thresh.set(str(cfg["sm_mod_threshold"]))

  def _refresh_sensor_tree(self, select_idx: Optional[int] = None) -> None:
    self._loading_sensor = True
    try:
      self.sensor_tree.delete(*self.sensor_tree.get_children())
      for idx, s in enumerate(self.sensors):
        self.sensor_tree.insert(
            "", tk.END, iid=str(idx), values=(s.sensor_id, s.distortions_summary())
        )
      if select_idx is not None and 0 <= select_idx < len(self.sensors):
        self.sensor_tree.selection_set(str(select_idx))
        self.selected_sensor_idx = select_idx
        self._load_sensor_into_editor(self.sensors[select_idx])
    finally:
      self._loading_sensor = False

  def _on_sensor_selected(self, _event: tk.Event) -> None:
    if self._loading_sensor:
      return
    sel = self.sensor_tree.selection()
    if not sel:
      return
    new_idx = int(sel[0])
    if self.selected_sensor_idx is not None and self.selected_sensor_idx != new_idx:
      self._commit_editor_to_sensor(self.selected_sensor_idx, silent=True)
      self._update_tree_row(self.selected_sensor_idx)
    self.selected_sensor_idx = new_idx
    self._load_sensor_into_editor(self.sensors[new_idx])

  def _update_tree_row(self, idx: int) -> None:
    if 0 <= idx < len(self.sensors):
      s = self.sensors[idx]
      self.sensor_tree.item(str(idx), values=(s.sensor_id, s.distortions_summary()))

  def _load_sensor_into_editor(self, s: SensorConfig) -> None:
    self.s_id_var.set(s.sensor_id)
    self.s_weight_var.set(str(s.manual_weight))
    self.s_gauss_on.set(s.gaussian_enabled)
    self.s_gauss_std.set(str(s.gaussian_std))
    self.s_gauss_seed.set(s.gaussian_seed)
    self.s_bias_on.set(s.bias_enabled)
    self.s_bias_val.set(str(s.bias_value))
    self.s_emi_on.set(s.emi_enabled)
    self.s_emi_prob.set(str(s.emi_prob))
    self.s_emi_amp.set(str(s.emi_amp))
    self.s_turb_on.set(s.turbulence_enabled)
    self.s_turb_scale.set(str(s.turbulence_scale))
    self.s_drift_on.set(s.drift_enabled)
    self.s_drift_int.set(str(s.drift_intensity))
    self.s_pwr_on.set(s.powerline_enabled)
    self.s_pwr_amp.set(str(s.powerline_amp))
    self.s_pwr_freq.set(str(s.powerline_freq))
    self.s_quant_on.set(s.quantization_enabled)
    self.s_quant_res.set(str(s.quantization_res))

  def _commit_editor_to_sensor(self, idx: int, silent: bool = False) -> bool:
    if not 0 <= idx < len(self.sensors):
      return True
    try:
      sensor_id = self.s_id_var.get().strip()
      if not sensor_id:
        raise ValueError("Sensor ID cannot be empty.")
      s = SensorConfig(
          sensor_id=sensor_id,
          manual_weight=float(self.s_weight_var.get()),
          gaussian_enabled=self.s_gauss_on.get(),
          gaussian_std=float(self.s_gauss_std.get()),
          gaussian_seed=self.s_gauss_seed.get().strip(),
          bias_enabled=self.s_bias_on.get(),
          bias_value=float(self.s_bias_val.get()),
          emi_enabled=self.s_emi_on.get(),
          emi_prob=float(self.s_emi_prob.get()),
          emi_amp=float(self.s_emi_amp.get()),
          turbulence_enabled=self.s_turb_on.get(),
          turbulence_scale=float(self.s_turb_scale.get()),
          drift_enabled=self.s_drift_on.get(),
          drift_intensity=float(self.s_drift_int.get()),
          powerline_enabled=self.s_pwr_on.get(),
          powerline_amp=float(self.s_pwr_amp.get()),
          powerline_freq=float(self.s_pwr_freq.get()),
          quantization_enabled=self.s_quant_on.get(),
          quantization_res=float(self.s_quant_res.get()),
      )
      if s.gaussian_seed:
        int(s.gaussian_seed)
      self.sensors[idx] = s
      self._update_tree_row(idx)
      return True
    except ValueError as exc:
      if not silent:
        messagebox.showerror("Invalid Sensor Configuration", str(exc))
      return False

  def _apply_sensor_edit(self) -> None:
    if self.selected_sensor_idx is not None:
      if self._commit_editor_to_sensor(self.selected_sensor_idx):
        self.status_var.set(f"Updated {self.sensors[self.selected_sensor_idx].sensor_id}.")

  def _add_sensor(self) -> None:
    if self.selected_sensor_idx is not None:
      self._commit_editor_to_sensor(self.selected_sensor_idx, silent=True)
    next_letter = chr(ord("A") + len(self.sensors))
    self.sensors.append(
        SensorConfig(
            sensor_id=f"Czujnik {next_letter}",
            gaussian_std=0.1,
            gaussian_seed=str(len(self.sensors) + 1),
        )
    )
    self._refresh_sensor_tree(select_idx=len(self.sensors) - 1)

  def _remove_sensor(self) -> None:
    if len(self.sensors) <= 1:
      messagebox.showwarning("Sensors", "At least one sensor is required.")
      return
    idx = self.selected_sensor_idx if self.selected_sensor_idx is not None else len(self.sensors) - 1
    self.sensors.pop(idx)
    new_idx = min(idx, len(self.sensors) - 1)
    self._refresh_sensor_tree(select_idx=new_idx)

  def _build_generator(self) -> SignalGenerator:
    params = {k: float(v.get()) for k, v in self.signal_param_vars.items()}
    sig_type = self.signal_type_var.get()
    if sig_type == "Sine Wave":
      return SineWaveGenerator(**params)
    if sig_type == "Square Wave":
      if params["t_max"] <= 0:
        raise ValueError("t_max must be > 0.")
      return SquareWaveGenerator(**params)
    if sig_type == "Exponential Response":
      if params["tau"] <= 0 or params["t_max"] <= 0:
        raise ValueError("tau and t_max must be > 0.")
      return ExponentialResponseGenerator(**params)
    raise ValueError(f"Unknown signal type: {sig_type}")

  def _build_voters(self) -> List[Voter]:
    voters: List[Voter] = []
    if self.v_weighted_on.get():
      if self.v_auto_weights.get():
        weights = []
        for s in self.sensors:
          if not s.gaussian_enabled or s.gaussian_std <= 0:
            raise ValueError(
                f"{s.sensor_id}: Auto weights (1/σ²) requires Gaussian noise with σ > 0."
            )
          weights.append(1.0 / (s.gaussian_std ** 2))
      else:
        weights = [s.manual_weight for s in self.sensors]
      voters.append(WeightedVoter(weights))

    if self.v_pc_on.get():
      voters.append(
          PredictorCorrectorVoter(
              agreement_threshold=float(self.v_pc_agree.get()),
              prediction_threshold=float(self.v_pc_pred.get()),
              prediction_method=self.v_pc_method.get(),
          )
      )

    if self.v_sm_on.get():
      voters.append(
          SmoothingVoter(
              agreement_threshold=float(self.v_sm_agree.get()),
              smoothing_threshold=float(self.v_sm_thresh.get()),
              accumulate_threshold=False,
          )
      )

    if self.v_sm_mod_on.get():
      voters.append(
          SmoothingVoter(
              agreement_threshold=float(self.v_sm_mod_agree.get()),
              smoothing_threshold=float(self.v_sm_mod_thresh.get()),
              accumulate_threshold=True,
          )
      )

    return voters

  def run_simulation(self) -> None:
    if self.selected_sensor_idx is not None:
      if not self._commit_editor_to_sensor(self.selected_sensor_idx):
        return

    try:
      num_samples = int(self.num_samples_var.get())
      if num_samples < 1:
        raise ValueError("Number of samples must be >= 1.")
      generator = self._build_generator()
      built_sensors = [s.build_sensor() for s in self.sensors]
      voters = self._build_voters()

      engine = SimulationEngine(generator, built_sensors, voters)
      result = engine.run(num_samples)
    except ValueError as exc:
      messagebox.showerror("Simulation Error", str(exc))
      self.status_var.set(f"Error: {exc}")
      return

    self.last_result = result
    ResultPlotter.plot_to_figure(result, self.figure)
    self.canvas.draw_idle()

    self.rmse_tree.delete(*self.rmse_tree.get_children())
    for name, err in result.rmse().items():
      valid_count = int(np.sum(~np.isnan(result.consensus[name])))
      rmse_str = f"{err:.4f}" if not np.isnan(err) else "N/A"
      self.rmse_tree.insert(
          "", tk.END, values=(name, rmse_str, f"{valid_count} / {num_samples}")
      )

    self.status_var.set(
        f"Simulation finished: {num_samples} samples, {len(built_sensors)} sensors, {len(voters)} voters."
    )

  def export_csv(self) -> None:
    if self.last_result is None:
      messagebox.showinfo("Export CSV", "Run a simulation first.")
      return
    path = filedialog.asksaveasfilename(
        title="Export Simulation Results to CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path:
      return
    CSVExporter.export(Path(path), self.last_result.to_columns())
    self.status_var.set(f"Exported CSV to {path}")

  def open_native_plot(self) -> None:
    """Otwiera wykres w natywnym oknie Matplotlib (pełna rozdzielczość Retina)."""
    if self.last_result is None:
      messagebox.showinfo("Native Plot", "Run a simulation first.")
      return

    res = self.last_result
    tmp_dir = Path(tempfile.mkdtemp(prefix="voting_plot_"))
    npz_path = tmp_dir / "result.npz"
    meta_path = tmp_dir / "meta.json"

    consensus_keys = list(res.consensus.keys())
    arrays = {
        "time": res.time,
        "reference": res.reference,
        "readings": res.readings,
        **{f"c_{i}": res.consensus[k] for i, k in enumerate(consensus_keys)},
    }
    np.savez(npz_path, **arrays)
    meta_path.write_text(
        json.dumps({
            "sensor_ids": res.sensor_ids,
            "consensus_keys": consensus_keys,
            "reference_name": res.reference_name,
        }),
        encoding="utf-8",
    )

    src_dir = str(Path(__file__).resolve().parent.parent)
    helper = (
        "import json, shutil, sys; from pathlib import Path; import numpy as np; "
        f"sys.path.insert(0, {src_dir!r}); "
        "from simulation.engine import SimulationResult; "
        "from simulation.plotters import ResultPlotter; "
        f"d = Path({str(tmp_dir)!r}); "
        "meta = json.loads((d / 'meta.json').read_text(encoding='utf-8')); "
        "data = np.load(d / 'result.npz'); "
        "consensus = {k: data[f'c_{i}'] for i, k in enumerate(meta['consensus_keys'])}; "
        "res = SimulationResult(time=data['time'], reference=data['reference'], "
        "sensor_ids=meta['sensor_ids'], readings=data['readings'], "
        "consensus=consensus, reference_name=meta['reference_name']); "
        "data.close(); shutil.rmtree(d, ignore_errors=True); "
        "ResultPlotter.plot(res, show=True)"
    )
    subprocess.Popen([sys.executable, "-c", helper])
    self.status_var.set("Opened plot in native Matplotlib window.")


def main() -> None:
  root = tk.Tk()
  VotingSimulationApp(root)
  root.mainloop()


if __name__ == "__main__":
  main()
