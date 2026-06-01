"""Estimate the energy use and carbon footprint of a computation.

We assess the environmental cost of the research project here. There is no GPU,
so the dominant cost is CPU time (plus a one-off data download). We estimate
energy from first principles so the calculation is transparent and reproducible:

    energy_kWh = runtime_hours * power_kW * PUE
    co2_grams  = energy_kWh * grid_carbon_intensity_gPerkWh

where

* ``power_kW``  -- average electrical power drawn while computing. We take a
  fraction of the CPU's thermal design power (TDP) plus a baseline for the
  rest of the machine.
* ``PUE``       -- Power Usage Effectiveness of the facility (data-centre
  overhead for cooling etc.). ~1.1 for an efficient cloud DC, ~1.6 for an
  average one, ~1.0 for a laptop on its own.
* grid carbon intensity -- gCO2 per kWh of the local electricity grid. This
  varies enormously (e.g. ~25 in Norway/France-nuclear, ~230 UK 2023 average,
  ~475 world average, ~700+ for coal-heavy grids).

This is the same methodology used by tools such as CodeCarbon and the ML CO2
Impact calculator (Lacoste et al., 2019); we implement it directly so the
project has no extra dependency.
"""
from __future__ import annotations

import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field, asdict


# Sensible defaults; override per machine / region.
DEFAULTS = dict(
    cpu_tdp_watts=65.0,        # typical desktop/server CPU package TDP
    cpu_utilisation=0.55,      # average fraction of TDP actually drawn
    baseline_watts=20.0,       # rest of the machine (RAM, board, disk)
    pue=1.2,                   # facility overhead
    grid_intensity=230.0,      # gCO2/kWh -- UK 2023 average
)

# Everyday equivalences (grams CO2 per unit) for putting numbers in context.
_EQUIV = dict(
    km_petrol_car=170.0,       # gCO2 per km, average petrol car
    smartphone_charge=8.3,     # gCO2 to fully charge a smartphone
    kettle_boil=70.0,          # gCO2 to boil a full kettle (UK grid)
)


@dataclass
class EnergyReport:
    seconds: float
    energy_kwh: float
    co2_grams: float
    params: dict = field(default_factory=dict)

    def equivalences(self) -> dict:
        return {
            "km_in_petrol_car": self.co2_grams / _EQUIV["km_petrol_car"],
            "smartphone_charges": self.co2_grams / _EQUIV["smartphone_charge"],
            "kettles_boiled": self.co2_grams / _EQUIV["kettle_boil"],
        }

    def summary(self) -> str:
        eq = self.equivalences()
        return (
            f"runtime      : {self.seconds:8.1f} s ({self.seconds/3600:.3f} h)\n"
            f"energy        : {self.energy_kwh*1000:8.2f} Wh ({self.energy_kwh:.5f} kWh)\n"
            f"CO2 emitted   : {self.co2_grams:8.2f} g\n"
            f"  ~= {eq['km_in_petrol_car']:.3f} km driven in a petrol car\n"
            f"  ~= {eq['smartphone_charges']:.2f} smartphone charges\n"
            f"  ~= {eq['kettles_boiled']:.3f} kettles boiled"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["equivalences"] = self.equivalences()
        return d


def estimate_energy(seconds: float, **overrides) -> EnergyReport:
    """Estimate energy + CO2 for a run of ``seconds`` wall-clock time."""
    p = {**DEFAULTS, **overrides}
    watts = p["cpu_tdp_watts"] * p["cpu_utilisation"] + p["baseline_watts"]
    power_kw = watts / 1000.0
    hours = seconds / 3600.0
    energy_kwh = power_kw * hours * p["pue"]
    co2 = energy_kwh * p["grid_intensity"]
    return EnergyReport(seconds=seconds, energy_kwh=energy_kwh, co2_grams=co2, params=p)


class track_energy(ContextDecorator):
    """Context manager that times a block and estimates its footprint.

    Example
    -------
    >>> with track_energy("train_cnn") as t:
    ...     train_cnn(...)
    >>> print(t.report.summary())
    """

    def __init__(self, label: str = "block", **overrides):
        self.label = label
        self.overrides = overrides
        self.report: EnergyReport | None = None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.report = estimate_energy(time.time() - self._t0, **self.overrides)
        return False
