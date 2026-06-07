"""Educational mean-field plasma response and threshold-logic model.

This module is intentionally not a laboratory control model.  It represents
ionization, recombination, excitation, conductivity, optical emission, and a
noisy detector with a small set of normalized ordinary differential equations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import erf, exp, sqrt
from typing import Iterable

import numpy as np
import pandas as pd


ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
STANDARD_PRESSURE_PA = 101_325.0
TOWNSEND_V_M2 = 1.0e-21


@dataclass(frozen=True)
class GasProperties:
    """Effective coefficients for the reduced educational model.

    The rates are effective first-order rates for normalized state variables,
    not cross-section data or validated reaction constants.
    """

    name: str
    ionization_threshold_td: float
    transition_width_td: float
    max_ionization_rate_s: float
    recombination_rate_s: float
    excitation_rate_s: float
    deexcitation_rate_s: float
    radiative_fraction: float
    electron_mobility_ref_m2_v_s: float
    electron_attachment_rate_s: float = 0.0
    quenching_strength: float = 1.0
    effective_electron_energy_threshold_ev: float = 3.2
    energy_transition_width_ev: float = 0.55
    metastable_production_rate_s: float = 650.0
    metastable_decay_rate_s: float = 180.0
    stepwise_ionization_rate_s: float = 420.0
    electron_detachment_rate_s: float = 100.0
    ion_ion_recombination_rate_s: float = 220.0
    positive_ion_mobility_ref_m2_v_s: float = 0.0020
    negative_ion_mobility_ref_m2_v_s: float = 0.0015


GAS_LIBRARY: dict[str, GasProperties] = {
    "argon": GasProperties(
        "argon", 115.0, 14.0, 4_500.0, 360.0, 3_200.0, 1_900.0, 0.72, 0.085,
        quenching_strength=0.8,
        effective_electron_energy_threshold_ev=3.0,
        metastable_production_rate_s=820.0,
        metastable_decay_rate_s=145.0,
        stepwise_ionization_rate_s=520.0,
    ),
    "neon": GasProperties(
        "neon", 150.0, 18.0, 4_100.0, 310.0, 3_800.0, 2_400.0, 0.82, 0.11,
        quenching_strength=0.65,
        effective_electron_energy_threshold_ev=3.4,
        metastable_production_rate_s=880.0,
        metastable_decay_rate_s=165.0,
        stepwise_ionization_rate_s=480.0,
    ),
    "helium": GasProperties(
        "helium", 185.0, 22.0, 3_700.0, 250.0, 3_500.0, 2_900.0, 0.58, 0.16,
        quenching_strength=0.45,
        effective_electron_energy_threshold_ev=3.8,
        metastable_production_rate_s=740.0,
        metastable_decay_rate_s=120.0,
        stepwise_ionization_rate_s=400.0,
    ),
    "air": GasProperties(
        "air", 135.0, 20.0, 3_700.0, 690.0, 2_500.0, 2_100.0, 0.38, 0.055,
        electron_attachment_rate_s=260.0,
        quenching_strength=1.5,
        effective_electron_energy_threshold_ev=3.1,
        metastable_production_rate_s=420.0,
        metastable_decay_rate_s=280.0,
        stepwise_ionization_rate_s=260.0,
        electron_detachment_rate_s=75.0,
        ion_ion_recombination_rate_s=390.0,
        negative_ion_mobility_ref_m2_v_s=0.0012,
    ),
}


@dataclass(frozen=True)
class PlasmaConfig:
    """User-facing physical and numerical inputs."""

    gas: str = "argon"
    pressure_pa: float = 4_000.0
    temperature_k: float = 300.0
    electric_field_v_m: float = 180_000.0
    applied_voltage_v: float = 900.0
    pulse_frequency_hz: float = 2_000.0
    pulse_width_s: float = 2.0e-4
    gas_flow_sccm: float = 50.0
    electrode_gap_m: float = 5.0e-3
    measurement_noise_std: float = 0.025
    initial_ionization_fraction: float = 1.0e-10
    binary_threshold: float = 0.45
    duration_s: float = 0.030
    dt_s: float = 2.0e-5
    random_seed: int = 2026

    # Model coefficients exposed for extension and sensitivity studies.
    field_voltage_blend: float = 0.50
    afterglow_field_fraction: float = 0.025
    max_ionization_fraction: float = 2.0e-7
    electron_wall_loss_rate_s: float = 300.0
    ion_wall_loss_rate_s: float = 85.0
    flow_loss_per_sccm_s: float = 0.55
    seed_floor_normalized: float = 1.0e-4
    density_ceiling_normalized: float = 1.5
    conductivity_reference_s_m: float = 0.020
    emission_reference_au: float = 0.30
    detector_time_constant_s: float = 7.5e-4
    optical_sensor_weight: float = 0.48


@dataclass
class SimulationResult:
    data: pd.DataFrame
    summary: dict[str, float | int | str]
    config: PlasmaConfig
    gas: GasProperties


def validate_config(config: PlasmaConfig) -> None:
    positive = {
        "pressure_pa": config.pressure_pa,
        "temperature_k": config.temperature_k,
        "electric_field_v_m": config.electric_field_v_m,
        "applied_voltage_v": config.applied_voltage_v,
        "pulse_frequency_hz": config.pulse_frequency_hz,
        "pulse_width_s": config.pulse_width_s,
        "electrode_gap_m": config.electrode_gap_m,
        "duration_s": config.duration_s,
        "dt_s": config.dt_s,
        "max_ionization_fraction": config.max_ionization_fraction,
        "detector_time_constant_s": config.detector_time_constant_s,
    }
    bad = [name for name, value in positive.items() if value <= 0.0]
    if bad:
        raise ValueError(f"These parameters must be positive: {', '.join(bad)}")
    if config.pulse_width_s > 1.0 / config.pulse_frequency_hz:
        raise ValueError("pulse_width_s cannot exceed one pulse period")
    if not 0.0 <= config.measurement_noise_std <= 1.0:
        raise ValueError("measurement_noise_std must be in [0, 1]")
    if not 0.0 < config.binary_threshold < 1.0:
        raise ValueError("binary_threshold must be in (0, 1)")
    if not 0.0 <= config.field_voltage_blend <= 1.0:
        raise ValueError("field_voltage_blend must be in [0, 1]")
    if not 0.0 <= config.optical_sensor_weight <= 1.0:
        raise ValueError("optical_sensor_weight must be in [0, 1]")
    if config.initial_ionization_fraction < 0.0:
        raise ValueError("initial_ionization_fraction cannot be negative")


def resolve_gas(
    config: PlasmaConfig, custom_gas: GasProperties | None = None
) -> GasProperties:
    if custom_gas is not None:
        return custom_gas
    key = config.gas.lower().strip()
    if key not in GAS_LIBRARY:
        available = ", ".join(sorted(GAS_LIBRARY))
        raise ValueError(
            f"Unknown gas '{config.gas}'. Use one of {available}, or pass custom_gas."
        )
    return GAS_LIBRARY[key]


def neutral_density_m3(config: PlasmaConfig) -> float:
    """Ideal-gas neutral number density N = p / (k_B T)."""

    return config.pressure_pa / (BOLTZMANN_J_K * config.temperature_k)


def blended_electric_field_v_m(config: PlasmaConfig) -> float:
    """Blend the explicit field with the voltage-derived field V/d.

    Both are requested inputs.  The blend avoids silently double-counting them.
    A value of 0 uses only electric_field_v_m and 1 uses only voltage/gap.
    """

    voltage_field = config.applied_voltage_v / config.electrode_gap_m
    blend = config.field_voltage_blend
    return (1.0 - blend) * config.electric_field_v_m + blend * voltage_field


def reduced_field_td(config: PlasmaConfig, instantaneous_field_v_m: float) -> float:
    """Return E/N in Townsend, where 1 Td = 1e-21 V m^2."""

    return instantaneous_field_v_m / neutral_density_m3(config) / TOWNSEND_V_M2


def logistic_activation(value: float, midpoint: float, width: float) -> float:
    z = np.clip((value - midpoint) / width, -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(-z)))


def _first_sustained_crossing(
    time_s: np.ndarray, binary: np.ndarray, hold_samples: int
) -> float:
    if hold_samples <= 1:
        hits = np.flatnonzero(binary)
        return float(time_s[hits[0]]) if len(hits) else float("nan")
    kernel = np.ones(hold_samples, dtype=int)
    sustained = np.convolve(binary.astype(int), kernel, mode="valid")
    hits = np.flatnonzero(sustained == hold_samples)
    return float(time_s[hits[0]]) if len(hits) else float("nan")


def _stability_metric(
    signal: np.ndarray, samples_per_period: int
) -> tuple[float, np.ndarray]:
    """Measure cycle-to-cycle convergence, not within-cycle pulse ripple."""

    n = len(signal)
    spp = max(2, samples_per_period)
    cycle_count = n // spp
    regimes = np.full(n, "transition", dtype=object)
    if cycle_count < 3:
        return 0.0, regimes

    cycle_means = np.array(
        [np.mean(signal[i * spp : (i + 1) * spp]) for i in range(cycle_count)]
    )
    tail = cycle_means[-min(8, cycle_count) :]
    scale = abs(float(np.mean(tail))) + 0.05
    coefficient_of_variation = float(np.std(tail) / scale)
    drift = float(abs(tail[-1] - tail[0]) / scale)
    metric = float(np.clip(exp(-4.0 * coefficient_of_variation - 2.0 * drift), 0, 1))

    for cycle_index in range(cycle_count):
        start = cycle_index * spp
        stop = min((cycle_index + 1) * spp, n)
        history = cycle_means[max(0, cycle_index - 3) : cycle_index + 1]
        mean_level = float(np.mean(history))
        local_scale = abs(mean_level) + 0.05
        local_cv = float(np.std(history) / local_scale)
        if mean_level < 0.02:
            label = "off"
        elif len(history) < 3:
            label = "transition"
        elif local_cv < 0.06:
            label = "stable"
        elif local_cv > 0.18:
            label = "unstable"
        else:
            label = "transition"
        regimes[start:stop] = label
    if cycle_count * spp < n:
        regimes[cycle_count * spp :] = regimes[cycle_count * spp - 1]
    return metric, regimes


def simulate_plasma(
    config: PlasmaConfig,
    drive_scale: float = 1.0,
    custom_gas: GasProperties | None = None,
) -> SimulationResult:
    """Integrate the reduced plasma model and return time series plus metrics."""

    validate_config(config)
    if drive_scale < 0.0:
        raise ValueError("drive_scale cannot be negative")
    gas = resolve_gas(config, custom_gas)
    rng = np.random.default_rng(config.random_seed)

    time_s = np.arange(0.0, config.duration_s + 0.5 * config.dt_s, config.dt_s)
    n_steps = len(time_s)
    period_s = 1.0 / config.pulse_frequency_hz
    pulse = (np.mod(time_s, period_s) < config.pulse_width_s).astype(float)
    base_field = blended_electric_field_v_m(config)
    n_neutral = neutral_density_m3(config)
    density_scale = config.max_ionization_fraction * n_neutral

    electron = np.zeros(n_steps)
    ion = np.zeros(n_steps)
    excitation = np.zeros(n_steps)
    detector_clean = np.zeros(n_steps)
    ionization_source_s = np.zeros(n_steps)
    recombination_source_s = np.zeros(n_steps)
    reduced_field = np.zeros(n_steps)
    activation = np.zeros(n_steps)

    initial_normalized = config.initial_ionization_fraction / max(
        config.max_ionization_fraction, 1e-30
    )
    electron[0] = np.clip(initial_normalized, 0.0, config.density_ceiling_normalized)
    ion[0] = electron[0]

    flow_loss = config.flow_loss_per_sccm_s * max(config.gas_flow_sccm, 0.0)
    electron_loss_rate = (
        config.electron_wall_loss_rate_s
        + gas.electron_attachment_rate_s
        + flow_loss
    )
    ion_loss_rate = config.ion_wall_loss_rate_s + 0.35 * flow_loss
    pressure_quench = 1.0 + gas.quenching_strength * (
        config.pressure_pa / STANDARD_PRESSURE_PA
    )
    effective_deexcitation_rate = gas.deexcitation_rate_s * pressure_quench

    # Euler integration is adequate because dt * max(rate) remains below ~0.1
    # for the supplied scenarios. State clipping prevents numerical negativity.
    for k in range(n_steps - 1):
        field_fraction = (
            drive_scale * pulse[k]
            + config.afterglow_field_fraction * (1.0 - pulse[k])
        )
        instantaneous_field = base_field * field_fraction
        reduced_field[k] = reduced_field_td(config, instantaneous_field)
        activation[k] = logistic_activation(
            reduced_field[k],
            gas.ionization_threshold_td,
            gas.transition_width_td,
        )

        available_capacity = max(
            0.0, 1.0 - electron[k] / config.density_ceiling_normalized
        )
        avalanche_population = electron[k] + config.seed_floor_normalized
        ionization_source_s[k] = (
            gas.max_ionization_rate_s
            * activation[k]
            * avalanche_population
            * available_capacity
        )
        recombination_source_s[k] = (
            gas.recombination_rate_s * electron[k] * ion[k]
        )

        d_electron = (
            ionization_source_s[k]
            - recombination_source_s[k]
            - electron_loss_rate * electron[k]
        )
        d_ion = (
            ionization_source_s[k]
            - recombination_source_s[k]
            - ion_loss_rate * ion[k]
        )
        excitation_drive = (
            gas.excitation_rate_s
            * activation[k]
            * electron[k]
            / (0.12 + electron[k])
            * (1.0 - excitation[k])
        )
        d_excitation = excitation_drive - effective_deexcitation_rate * excitation[k]

        electron[k + 1] = np.clip(
            electron[k] + config.dt_s * d_electron,
            0.0,
            config.density_ceiling_normalized,
        )
        ion[k + 1] = np.clip(
            ion[k] + config.dt_s * d_ion,
            0.0,
            config.density_ceiling_normalized,
        )
        excitation[k + 1] = np.clip(
            excitation[k] + config.dt_s * d_excitation, 0.0, 1.0
        )

        mobility = min(
            20.0,
            gas.electron_mobility_ref_m2_v_s
            * STANDARD_PRESSURE_PA
            / config.pressure_pa
            * sqrt(config.temperature_k / 300.0),
        )
        conductivity = (
            ELEMENTARY_CHARGE_C * electron[k] * density_scale * mobility
        )
        emission = (
            gas.radiative_fraction
            * effective_deexcitation_rate
            * excitation[k]
            / 1_000.0
        )
        conductivity_response = conductivity / (
            conductivity + config.conductivity_reference_s_m
        )
        emission_response = emission / (emission + config.emission_reference_au)
        sensor_raw = (
            (1.0 - config.optical_sensor_weight) * conductivity_response
            + config.optical_sensor_weight * emission_response
        )
        detector_clean[k + 1] = detector_clean[k] + config.dt_s * (
            sensor_raw - detector_clean[k]
        ) / config.detector_time_constant_s

    reduced_field[-1] = reduced_field[-2]
    activation[-1] = activation[-2]
    ionization_source_s[-1] = ionization_source_s[-2]
    recombination_source_s[-1] = recombination_source_s[-2]

    mobility = min(
        20.0,
        gas.electron_mobility_ref_m2_v_s
        * STANDARD_PRESSURE_PA
        / config.pressure_pa
        * sqrt(config.temperature_k / 300.0),
    )
    electron_density = electron * density_scale
    ion_density = ion * density_scale
    conductivity = ELEMENTARY_CHARGE_C * electron_density * mobility
    emission = (
        gas.radiative_fraction
        * effective_deexcitation_rate
        * excitation
        / 1_000.0
    )
    measured_sensor = np.clip(
        detector_clean
        + rng.normal(0.0, config.measurement_noise_std, size=n_steps),
        0.0,
        1.0,
    )
    clean_binary = (detector_clean >= config.binary_threshold).astype(int)
    binary = (measured_sensor >= config.binary_threshold).astype(int)

    samples_per_period = max(2, int(round(period_s / config.dt_s)))
    stability, regimes = _stability_metric(detector_clean, samples_per_period)
    tail_start = int(0.75 * n_steps)
    noise_robustness = float(np.mean(binary[tail_start:] == clean_binary[tail_start:]))
    hold_samples = max(2, int(round(0.20 * period_s / config.dt_s)))
    delay_s = _first_sustained_crossing(time_s, clean_binary, hold_samples)
    tail_mean = float(np.mean(detector_clean[tail_start:]))
    tail_binary = int(tail_mean >= config.binary_threshold)
    margin = abs(tail_mean - config.binary_threshold)
    if config.measurement_noise_std > 0.0:
        z = margin / config.measurement_noise_std
        theoretical_noise_confidence = float(
            np.clip(2.0 * (0.5 * (1.0 + erf(z / sqrt(2.0)))) - 1.0, 0.0, 1.0)
        )
    else:
        theoretical_noise_confidence = 1.0

    data = pd.DataFrame(
        {
            "time_s": time_s,
            "pulse_on": pulse.astype(int),
            "reduced_field_td": reduced_field,
            "ionization_activation": activation,
            "ionization_source_normalized_s": ionization_source_s,
            "recombination_source_normalized_s": recombination_source_s,
            "ionization_fraction": electron * config.max_ionization_fraction,
            "electron_density_m3": electron_density,
            "ion_density_m3": ion_density,
            "excitation_fraction": excitation,
            "conductivity_s_m": conductivity,
            "emission_intensity_au": emission,
            "sensor_clean": detector_clean,
            "sensor_measured": measured_sensor,
            "binary_output": binary,
            "logic_output": binary,
            "regime": regimes,
        }
    )
    final_regime = str(pd.Series(regimes[tail_start:]).mode().iloc[0])
    summary: dict[str, float | int | str] = {
        "gas": gas.name,
        "neutral_density_m3": n_neutral,
        "blended_field_v_m": base_field,
        "base_reduced_field_td": reduced_field_td(config, base_field),
        "drive_scale": drive_scale,
        "tail_sensor_mean": tail_mean,
        "tail_binary_output": tail_binary,
        "delay_s": delay_s,
        "stability_metric": stability,
        "noise_robustness_metric": noise_robustness,
        "theoretical_noise_confidence": theoretical_noise_confidence,
        "final_regime": final_regime,
        "peak_electron_density_m3": float(np.max(electron_density)),
        "peak_ionization_fraction": float(
            np.max(electron * config.max_ionization_fraction)
        ),
        "peak_emission_au": float(np.max(emission)),
        "peak_conductivity_s_m": float(np.max(conductivity)),
    }
    return SimulationResult(data=data, summary=summary, config=config, gas=gas)


def logic_truth_tables(
    config: PlasmaConfig,
    base_drive: float = 0.25,
    per_active_input_drive: float = 0.35,
    custom_gas: GasProperties | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Calibrate physical response levels and derive threshold logic tables.

    OR uses a threshold between zero- and one-input response. AND uses a
    threshold between one- and two-input response. XOR uses the window between
    those thresholds. NAND and NOR invert the corresponding detector decision.
    """

    responses: dict[int, float] = {}
    for active_count in (0, 1, 2):
        response = simulate_plasma(
            config,
            drive_scale=base_drive + per_active_input_drive * active_count,
            custom_gas=custom_gas,
        )
        responses[active_count] = float(response.summary["tail_sensor_mean"])

    or_threshold = 0.5 * (responses[0] + responses[1])
    and_threshold = 0.5 * (responses[1] + responses[2])
    if not responses[0] < responses[1] < responses[2]:
        raise RuntimeError(
            "Logic calibration requires monotonically increasing physical response; "
            f"got {responses}."
        )

    rows: list[dict[str, float | int]] = []
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        sensor = responses[a + b]
        or_output = int(sensor >= or_threshold)
        and_output = int(sensor >= and_threshold)
        rows.append(
            {
                "A": a,
                "B": b,
                "active_input_count": a + b,
                "sensor_response": sensor,
                "OR": or_output,
                "AND": and_output,
                "NAND": 1 - and_output,
                "NOR": 1 - or_output,
                "XOR": int(or_threshold <= sensor < and_threshold),
                "HALF_ADDER_SUM": int(or_threshold <= sensor < and_threshold),
                "HALF_ADDER_CARRY": and_output,
            }
        )
    binary_table = pd.DataFrame(rows)

    not_threshold = 0.5 * (responses[0] + responses[1])
    not_rows = []
    for a in (0, 1):
        # The input is modeled as a suppressing input: A=1 reduces the drive.
        sensor = responses[1 - a]
        not_rows.append(
            {
                "A": a,
                "effective_drive_state": 1 - a,
                "sensor_response": sensor,
                "NOT": int(sensor >= not_threshold),
            }
        )
    not_table = pd.DataFrame(not_rows)
    calibration = {
        "response_0_active": responses[0],
        "response_1_active": responses[1],
        "response_2_active": responses[2],
        "or_threshold": or_threshold,
        "and_threshold": and_threshold,
        "not_threshold": not_threshold,
    }
    return binary_table, not_table, calibration


def full_adder_truth_table(
    config: PlasmaConfig,
    base_drive: float = 0.50,
    per_active_input_drive: float = 0.10,
    custom_gas: GasProperties | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Derive a three-input full adder from four calibrated response levels.

    The analog response is assigned to one of four adjacent windows
    representing zero, one, two, or three active physical inputs. The sum bit
    is the parity of that inferred count and carry is high for counts >= 2.
    """

    responses: dict[int, float] = {}
    for active_count in range(4):
        result = simulate_plasma(
            config,
            drive_scale=base_drive + per_active_input_drive * active_count,
            custom_gas=custom_gas,
        )
        responses[active_count] = float(result.summary["tail_sensor_mean"])
    levels = np.array([responses[index] for index in range(4)])
    if not np.all(np.diff(levels) > 0.0):
        raise RuntimeError(
            "Full-adder calibration requires four increasing response levels; "
            f"got {responses}."
        )
    boundaries = 0.5 * (levels[:-1] + levels[1:])
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            for carry_in in (0, 1):
                active_count = a + b + carry_in
                sensor = responses[active_count]
                inferred_count = int(np.searchsorted(boundaries, sensor, side="right"))
                rows.append(
                    {
                        "A": a,
                        "B": b,
                        "CARRY_IN": carry_in,
                        "active_input_count": active_count,
                        "sensor_response": sensor,
                        "inferred_active_count": inferred_count,
                        "SUM": inferred_count % 2,
                        "CARRY_OUT": int(inferred_count >= 2),
                    }
                )
    calibration = {
        **{f"response_{index}_active": responses[index] for index in range(4)},
        **{
            f"boundary_{index}_{index + 1}": float(boundaries[index])
            for index in range(3)
        },
    }
    return pd.DataFrame(rows), calibration


def schmitt_trigger(
    signal: Iterable[float],
    lower_threshold: float,
    upper_threshold: float,
    initial_state: int = 0,
) -> np.ndarray:
    """Stateful hysteresis threshold that suppresses near-threshold chatter."""

    if not lower_threshold < upper_threshold:
        raise ValueError("lower_threshold must be smaller than upper_threshold")
    if initial_state not in (0, 1):
        raise ValueError("initial_state must be 0 or 1")
    values = np.asarray(list(signal), dtype=float)
    output = np.empty(len(values), dtype=int)
    state = initial_state
    for index, value in enumerate(values):
        if state == 0 and value >= upper_threshold:
            state = 1
        elif state == 1 and value <= lower_threshold:
            state = 0
        output[index] = state
    return output


def sr_latch_sequence(
    set_input: Iterable[int],
    reset_input: Iterable[int],
    initial_state: int = 0,
) -> pd.DataFrame:
    """Evaluate an active-high conceptual SR latch input sequence.

    Simultaneous set and reset is marked invalid and leaves the stored state
    unchanged so subsequent educational analysis remains deterministic.
    """

    set_values = np.asarray(list(set_input), dtype=int)
    reset_values = np.asarray(list(reset_input), dtype=int)
    if len(set_values) != len(reset_values):
        raise ValueError("set_input and reset_input must have equal lengths")
    if not np.isin(set_values, [0, 1]).all() or not np.isin(
        reset_values, [0, 1]
    ).all():
        raise ValueError("latch inputs must contain only 0 and 1")
    if initial_state not in (0, 1):
        raise ValueError("initial_state must be 0 or 1")

    state = initial_state
    rows = []
    for step, (set_value, reset_value) in enumerate(
        zip(set_values, reset_values, strict=True)
    ):
        invalid = int(set_value == 1 and reset_value == 1)
        if not invalid:
            if set_value:
                state = 1
            elif reset_value:
                state = 0
        rows.append(
            {
                "step": step,
                "SET": int(set_value),
                "RESET": int(reset_value),
                "Q": state,
                "Q_BAR": 1 - state,
                "invalid_input": invalid,
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    result: SimulationResult, thresholds: Iterable[float]
) -> pd.DataFrame:
    rows = []
    signal = result.data["sensor_clean"].to_numpy()
    time_s = result.data["time_s"].to_numpy()
    tail_start = int(0.75 * len(signal))
    period_s = 1.0 / result.config.pulse_frequency_hz
    hold_samples = max(2, int(round(0.20 * period_s / result.config.dt_s)))
    for threshold in thresholds:
        binary = (signal >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "tail_high_fraction": float(np.mean(binary[tail_start:])),
                "tail_mean_output": int(np.mean(signal[tail_start:]) >= threshold),
                "delay_s": _first_sustained_crossing(
                    time_s, binary, hold_samples
                ),
                "tail_noise_margin": float(
                    abs(np.mean(signal[tail_start:]) - threshold)
                ),
            }
        )
    return pd.DataFrame(rows)


def parameter_sensitivity(
    config: PlasmaConfig,
    relative_step: float = 0.10,
    parameters: Iterable[str] | None = None,
    custom_gas: GasProperties | None = None,
) -> pd.DataFrame:
    """One-at-a-time normalized sensitivity of the tail sensor response."""

    if not 0.0 < relative_step < 1.0:
        raise ValueError("relative_step must be in (0, 1)")
    names = list(
        parameters
        or (
            "pressure_pa",
            "temperature_k",
            "electric_field_v_m",
            "applied_voltage_v",
            "pulse_frequency_hz",
            "pulse_width_s",
            "gas_flow_sccm",
            "electrode_gap_m",
            "initial_ionization_fraction",
        )
    )
    baseline = simulate_plasma(config, custom_gas=custom_gas)
    y0 = float(baseline.summary["tail_sensor_mean"])
    rows = []
    for name in names:
        x0 = float(getattr(config, name))
        if x0 == 0.0:
            continue
        lower_value = x0 * (1.0 - relative_step)
        upper_value = x0 * (1.0 + relative_step)
        lower = replace(config, **{name: lower_value})
        upper = replace(config, **{name: upper_value})
        try:
            y_lower = float(
                simulate_plasma(lower, custom_gas=custom_gas).summary[
                    "tail_sensor_mean"
                ]
            )
            y_upper = float(
                simulate_plasma(upper, custom_gas=custom_gas).summary[
                    "tail_sensor_mean"
                ]
            )
        except ValueError:
            # Pulse width may become greater than the shortened period.
            rows.append(
                {
                    "parameter": name,
                    "baseline_value": x0,
                    "lower_response": float("nan"),
                    "upper_response": float("nan"),
                    "normalized_sensitivity": float("nan"),
                    "absolute_sensitivity": float("nan"),
                }
            )
            continue
        normalized = (y_upper - y_lower) / (
            2.0 * relative_step * max(abs(y0), 1.0e-9)
        )
        rows.append(
            {
                "parameter": name,
                "baseline_value": x0,
                "lower_response": y_lower,
                "upper_response": y_upper,
                "normalized_sensitivity": normalized,
                "absolute_sensitivity": abs(normalized),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "absolute_sensitivity", ascending=False, na_position="last"
    )


def config_as_dict(config: PlasmaConfig) -> dict[str, object]:
    return asdict(config)
