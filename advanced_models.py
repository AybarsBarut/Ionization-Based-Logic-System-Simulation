"""Advanced educational extensions for the ionization-based logic model.

The extensions remain conceptual: adaptive ODE integration, electron energy,
negative ions, metastables, uncertainty propagation, coefficient fitting, and
a one-dimensional reaction-diffusion demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import erf, exp, sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from plasma_model import (
    ELEMENTARY_CHARGE_C,
    STANDARD_PRESSURE_PA,
    GasProperties,
    PlasmaConfig,
    SimulationResult,
    _first_sustained_crossing,
    _stability_metric,
    blended_electric_field_v_m,
    logistic_activation,
    neutral_density_m3,
    reduced_field_td,
    resolve_gas,
    simulate_plasma,
    validate_config,
)


@dataclass(frozen=True)
class AdvancedModelOptions:
    solver_method: str = "BDF"
    relative_tolerance: float = 1.0e-6
    absolute_tolerance: float = 1.0e-9
    initial_electron_energy_ev: float = 0.7
    background_electron_energy_ev: float = 0.45
    field_energy_gain_ev_per_td: float = 0.019
    maximum_electron_energy_ev: float = 10.0
    energy_relaxation_rate_s: float = 5_500.0
    excitation_energy_loss_ev: float = 0.35
    ionization_energy_loss_ev: float = 0.55
    negative_ion_wall_loss_rate_s: float = 65.0
    metastable_quenching_rate_s: float = 90.0


@dataclass
class SpatialSimulationResult:
    data: pd.DataFrame
    summary: dict[str, float | int | str]
    config: PlasmaConfig
    spatial_config: "SpatialModelConfig"
    gas: GasProperties


@dataclass(frozen=True)
class SpatialModelConfig:
    length_m: float = 5.0e-3
    grid_points: int = 61
    duration_s: float = 0.012
    dt_s: float = 5.0e-6
    snapshot_interval_s: float = 5.0e-4
    electron_diffusion_m2_s: float = 2.0e-4
    positive_ion_diffusion_m2_s: float = 2.5e-5
    negative_ion_diffusion_m2_s: float = 1.8e-5
    metastable_diffusion_m2_s: float = 6.0e-5
    field_edge_fraction: float = 0.72
    initial_perturbation_fraction: float = 0.03


@dataclass
class CalibrationResult:
    calibrated_gas: GasProperties
    comparison: pd.DataFrame
    diagnostics: dict[str, float | int | str]


def _mobility_scale(config: PlasmaConfig) -> float:
    return STANDARD_PRESSURE_PA / config.pressure_pa * sqrt(
        config.temperature_k / 300.0
    )


def _advanced_rates(
    time_s: float,
    state: np.ndarray,
    config: PlasmaConfig,
    gas: GasProperties,
    options: AdvancedModelOptions,
    drive_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    electron, positive_ion, negative_ion, excitation, metastable, energy_ev, detector = (
        np.maximum(state, 0.0)
    )
    period_s = 1.0 / config.pulse_frequency_hz
    pulse_on = float((time_s % period_s) < config.pulse_width_s)
    field_fraction = (
        drive_scale * pulse_on
        + config.afterglow_field_fraction * (1.0 - pulse_on)
    )
    instantaneous_field = blended_electric_field_v_m(config) * field_fraction
    field_td = reduced_field_td(config, instantaneous_field)
    field_activation = logistic_activation(
        field_td, gas.ionization_threshold_td, gas.transition_width_td
    )
    energy_activation = logistic_activation(
        energy_ev,
        gas.effective_electron_energy_threshold_ev,
        gas.energy_transition_width_ev,
    )
    activation = sqrt(max(field_activation * energy_activation, 0.0))

    capacity = max(
        0.0,
        1.0
        - (electron + negative_ion) / config.density_ceiling_normalized,
    )
    population = electron + config.seed_floor_normalized
    direct_ionization = (
        gas.max_ionization_rate_s * activation * population * capacity
    )
    stepwise_ionization = (
        gas.stepwise_ionization_rate_s
        * metastable
        * population
        * capacity
    )
    attachment = (
        gas.electron_attachment_rate_s
        * (1.0 - 0.65 * field_activation)
        * electron
    )
    detachment = (
        gas.electron_detachment_rate_s * field_activation * negative_ion
    )
    electron_ion_recombination = (
        gas.recombination_rate_s * electron * positive_ion
    )
    ion_ion_recombination = (
        gas.ion_ion_recombination_rate_s * positive_ion * negative_ion
    )
    flow_loss = config.flow_loss_per_sccm_s * max(config.gas_flow_sccm, 0.0)
    electron_wall_loss = config.electron_wall_loss_rate_s + flow_loss
    positive_ion_loss = config.ion_wall_loss_rate_s + 0.35 * flow_loss
    negative_ion_loss = options.negative_ion_wall_loss_rate_s + 0.25 * flow_loss

    pressure_quench = 1.0 + gas.quenching_strength * (
        config.pressure_pa / STANDARD_PRESSURE_PA
    )
    deexcitation_rate = gas.deexcitation_rate_s * pressure_quench
    excitation_drive = (
        gas.excitation_rate_s
        * activation
        * electron
        / (0.12 + electron)
        * (1.0 - excitation)
    )
    metastable_drive = (
        gas.metastable_production_rate_s
        * activation
        * electron
        / (0.10 + electron)
        * (1.0 - metastable)
    )
    metastable_loss_rate = (
        gas.metastable_decay_rate_s
        + options.metastable_quenching_rate_s
        * config.pressure_pa
        / STANDARD_PRESSURE_PA
    )

    target_energy = min(
        options.maximum_electron_energy_ev,
        options.background_electron_energy_ev
        + options.field_energy_gain_ev_per_td * field_td,
    )
    inelastic_loss = (
        options.excitation_energy_loss_ev * excitation_drive
        + options.ionization_energy_loss_ev
        * (direct_ionization + stepwise_ionization)
        / max(population, 1.0e-8)
    )
    energy_rate = (
        options.energy_relaxation_rate_s * (target_energy - energy_ev)
        - inelastic_loss
    )

    d_electron = (
        direct_ionization
        + stepwise_ionization
        + detachment
        - attachment
        - electron_ion_recombination
        - electron_wall_loss * electron
    )
    d_positive = (
        direct_ionization
        + stepwise_ionization
        - electron_ion_recombination
        - ion_ion_recombination
        - positive_ion_loss * positive_ion
    )
    d_negative = (
        attachment
        - detachment
        - ion_ion_recombination
        - negative_ion_loss * negative_ion
    )
    d_excitation = excitation_drive - deexcitation_rate * excitation
    d_metastable = (
        metastable_drive
        - metastable_loss_rate * metastable
        - stepwise_ionization
    )

    density_scale = config.max_ionization_fraction * neutral_density_m3(config)
    mobility_scale = _mobility_scale(config)
    electron_mobility = min(
        20.0, gas.electron_mobility_ref_m2_v_s * mobility_scale
    )
    positive_mobility = gas.positive_ion_mobility_ref_m2_v_s * mobility_scale
    negative_mobility = gas.negative_ion_mobility_ref_m2_v_s * mobility_scale
    conductivity = ELEMENTARY_CHARGE_C * density_scale * (
        electron * electron_mobility
        + positive_ion * positive_mobility
        + negative_ion * negative_mobility
    )
    emission = (
        gas.radiative_fraction
        * deexcitation_rate
        * excitation
        / 1_000.0
        + 0.08 * gas.radiative_fraction * metastable
    )
    conductivity_response = conductivity / (
        conductivity + config.conductivity_reference_s_m
    )
    emission_response = emission / (emission + config.emission_reference_au)
    sensor_raw = (
        (1.0 - config.optical_sensor_weight) * conductivity_response
        + config.optical_sensor_weight * emission_response
    )
    d_detector = (sensor_raw - detector) / config.detector_time_constant_s

    rates = np.array(
        [
            d_electron,
            d_positive,
            d_negative,
            d_excitation,
            d_metastable,
            energy_rate,
            d_detector,
        ]
    )
    diagnostics = {
        "pulse_on": pulse_on,
        "reduced_field_td": field_td,
        "field_activation": field_activation,
        "energy_activation": energy_activation,
        "ionization_activation": activation,
        "direct_ionization_source_s": direct_ionization,
        "stepwise_ionization_source_s": stepwise_ionization,
        "attachment_source_s": attachment,
        "detachment_source_s": detachment,
        "recombination_source_s": electron_ion_recombination,
        "conductivity_s_m": conductivity,
        "emission_intensity_au": emission,
        "sensor_raw": sensor_raw,
    }
    return rates, diagnostics


def simulate_advanced_plasma(
    config: PlasmaConfig,
    drive_scale: float = 1.0,
    custom_gas: GasProperties | None = None,
    options: AdvancedModelOptions | None = None,
) -> SimulationResult:
    """Solve the extended zero-dimensional model with an adaptive ODE method."""

    validate_config(config)
    if drive_scale < 0.0:
        raise ValueError("drive_scale cannot be negative")
    options = options or AdvancedModelOptions()
    if options.solver_method not in {"BDF", "Radau", "RK45", "LSODA"}:
        raise ValueError("Unsupported solver_method")
    gas = resolve_gas(config, custom_gas)
    output_steps = max(1, int(round(config.duration_s / config.dt_s)))
    time_s = np.linspace(0.0, config.duration_s, output_steps + 1)
    initial_electron = np.clip(
        config.initial_ionization_fraction
        / max(config.max_ionization_fraction, 1.0e-30),
        0.0,
        config.density_ceiling_normalized,
    )
    initial_state = np.array(
        [
            initial_electron,
            initial_electron,
            0.0,
            0.0,
            0.0,
            options.initial_electron_energy_ev,
            0.0,
        ]
    )

    def rhs(time_value: float, state: np.ndarray) -> np.ndarray:
        rates, _ = _advanced_rates(
            time_value, state, config, gas, options, drive_scale
        )
        return rates

    solution = solve_ivp(
        rhs,
        (0.0, config.duration_s),
        initial_state,
        method=options.solver_method,
        t_eval=time_s,
        max_step=config.dt_s,
        rtol=options.relative_tolerance,
        atol=options.absolute_tolerance,
    )
    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    states = np.maximum(solution.y.T, 0.0)
    states[:, :5] = np.minimum(
        states[:, :5], config.density_ceiling_normalized
    )
    states[:, 5] = np.minimum(
        states[:, 5], options.maximum_electron_energy_ev
    )
    states[:, 6] = np.clip(states[:, 6], 0.0, 1.0)

    diagnostics = []
    for time_value, state in zip(time_s, states, strict=True):
        _, values = _advanced_rates(
            float(time_value), state, config, gas, options, drive_scale
        )
        diagnostics.append(values)
    diagnostic_frame = pd.DataFrame(diagnostics)

    electron = states[:, 0]
    positive_ion = states[:, 1]
    negative_ion = states[:, 2]
    excitation = states[:, 3]
    metastable = states[:, 4]
    energy_ev = states[:, 5]
    detector_clean = states[:, 6]
    n_neutral = neutral_density_m3(config)
    density_scale = config.max_ionization_fraction * n_neutral
    rng = np.random.default_rng(config.random_seed)
    measured_sensor = np.clip(
        detector_clean
        + rng.normal(0.0, config.measurement_noise_std, size=len(time_s)),
        0.0,
        1.0,
    )
    clean_binary = (detector_clean >= config.binary_threshold).astype(int)
    binary = (measured_sensor >= config.binary_threshold).astype(int)
    period_s = 1.0 / config.pulse_frequency_hz
    samples_per_period = max(2, int(round(period_s / config.dt_s)))
    stability, regimes = _stability_metric(detector_clean, samples_per_period)
    tail_start = int(0.75 * len(time_s))
    hold_samples = max(2, int(round(0.20 * period_s / config.dt_s)))
    delay_s = _first_sustained_crossing(time_s, clean_binary, hold_samples)
    tail_mean = float(np.mean(detector_clean[tail_start:]))
    noise_robustness = float(
        np.mean(binary[tail_start:] == clean_binary[tail_start:])
    )
    margin = abs(tail_mean - config.binary_threshold)
    if config.measurement_noise_std > 0.0:
        z_value = margin / config.measurement_noise_std
        noise_confidence = float(
            np.clip(
                2.0 * (0.5 * (1.0 + erf(z_value / sqrt(2.0)))) - 1.0,
                0.0,
                1.0,
            )
        )
    else:
        noise_confidence = 1.0

    data = pd.DataFrame(
        {
            "time_s": time_s,
            "pulse_on": diagnostic_frame["pulse_on"].astype(int),
            "reduced_field_td": diagnostic_frame["reduced_field_td"],
            "field_activation": diagnostic_frame["field_activation"],
            "energy_activation": diagnostic_frame["energy_activation"],
            "ionization_activation": diagnostic_frame["ionization_activation"],
            "direct_ionization_source_normalized_s": diagnostic_frame[
                "direct_ionization_source_s"
            ],
            "stepwise_ionization_source_normalized_s": diagnostic_frame[
                "stepwise_ionization_source_s"
            ],
            "attachment_source_normalized_s": diagnostic_frame[
                "attachment_source_s"
            ],
            "detachment_source_normalized_s": diagnostic_frame[
                "detachment_source_s"
            ],
            "recombination_source_normalized_s": diagnostic_frame[
                "recombination_source_s"
            ],
            "ionization_fraction": electron * config.max_ionization_fraction,
            "electron_density_m3": electron * density_scale,
            "ion_density_m3": positive_ion * density_scale,
            "negative_ion_density_m3": negative_ion * density_scale,
            "excitation_fraction": excitation,
            "metastable_fraction": metastable,
            "electron_energy_ev": energy_ev,
            "electron_temperature_k": energy_ev * 11_604.518,
            "conductivity_s_m": diagnostic_frame["conductivity_s_m"],
            "emission_intensity_au": diagnostic_frame["emission_intensity_au"],
            "sensor_raw": diagnostic_frame["sensor_raw"],
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
        "solver_method": options.solver_method,
        "solver_function_evaluations": int(solution.nfev),
        "solver_jacobian_evaluations": int(getattr(solution, "njev", 0)),
        "neutral_density_m3": n_neutral,
        "base_reduced_field_td": reduced_field_td(
            config, blended_electric_field_v_m(config)
        ),
        "drive_scale": drive_scale,
        "tail_sensor_mean": tail_mean,
        "tail_binary_output": int(tail_mean >= config.binary_threshold),
        "delay_s": delay_s,
        "stability_metric": stability,
        "noise_robustness_metric": noise_robustness,
        "theoretical_noise_confidence": noise_confidence,
        "final_regime": final_regime,
        "peak_electron_density_m3": float(np.max(electron * density_scale)),
        "peak_negative_ion_density_m3": float(
            np.max(negative_ion * density_scale)
        ),
        "peak_metastable_fraction": float(np.max(metastable)),
        "mean_tail_electron_energy_ev": float(np.mean(energy_ev[tail_start:])),
        "peak_emission_au": float(
            np.max(diagnostic_frame["emission_intensity_au"])
        ),
        "peak_conductivity_s_m": float(
            np.max(diagnostic_frame["conductivity_s_m"])
        ),
    }
    return SimulationResult(data=data, summary=summary, config=config, gas=gas)


def monte_carlo_uncertainty(
    config: PlasmaConfig,
    sample_count: int = 80,
    relative_standard_deviation: float = 0.08,
    random_seed: int = 2026,
    use_advanced_model: bool = False,
    custom_gas: GasProperties | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate independent log-normal input uncertainty to model outputs."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if relative_standard_deviation <= 0.0:
        raise ValueError("relative_standard_deviation must be positive")
    rng = np.random.default_rng(random_seed)
    variable_names = (
        "pressure_pa",
        "temperature_k",
        "electric_field_v_m",
        "applied_voltage_v",
        "pulse_width_s",
        "gas_flow_sccm",
        "initial_ionization_fraction",
    )
    rows = []
    sigma_log = sqrt(np.log1p(relative_standard_deviation**2))
    for sample_index in range(sample_count):
        sampled_values = {}
        for name in variable_names:
            baseline = float(getattr(config, name))
            multiplier = float(
                np.exp(rng.normal(-0.5 * sigma_log**2, sigma_log))
            )
            sampled_values[name] = baseline * multiplier
        maximum_width = 0.95 / config.pulse_frequency_hz
        sampled_values["pulse_width_s"] = min(
            sampled_values["pulse_width_s"], maximum_width
        )
        sampled = replace(
            config,
            **sampled_values,
            random_seed=random_seed + sample_index,
        )
        if use_advanced_model:
            result = simulate_advanced_plasma(
                sampled, custom_gas=custom_gas
            )
        else:
            result = simulate_plasma(sampled, custom_gas=custom_gas)
        rows.append(
            {
                "sample": sample_index,
                **sampled_values,
                "tail_sensor_mean": result.summary["tail_sensor_mean"],
                "binary_output": result.summary["tail_binary_output"],
                "stability_metric": result.summary["stability_metric"],
                "delay_s": result.summary["delay_s"],
            }
        )
    samples = pd.DataFrame(rows)
    sensor = samples["tail_sensor_mean"]
    summary = pd.DataFrame(
        [
            {
                "sample_count": sample_count,
                "relative_input_std": relative_standard_deviation,
                "sensor_mean": float(sensor.mean()),
                "sensor_std": float(sensor.std(ddof=1)),
                "sensor_q025": float(sensor.quantile(0.025)),
                "sensor_q50": float(sensor.quantile(0.50)),
                "sensor_q975": float(sensor.quantile(0.975)),
                "probability_output_1": float(samples["binary_output"].mean()),
                "mean_stability": float(samples["stability_metric"].mean()),
            }
        ]
    )
    return samples, summary


def calibrate_gas_to_observations(
    config: PlasmaConfig,
    base_gas: GasProperties,
    observations: pd.DataFrame,
    parameter_names: Iterable[str] = (
        "ionization_threshold_td",
        "max_ionization_rate_s",
        "recombination_rate_s",
    ),
    maximum_function_evaluations: int = 60,
) -> CalibrationResult:
    """Fit selected positive gas coefficients to CSV-ready sensor observations.

    Required columns are ``drive_scale`` and ``sensor_observed``. An optional
    ``weight`` column controls residual weighting.
    """

    required = {"drive_scale", "sensor_observed"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Missing observation columns: {sorted(missing)}")
    names = tuple(parameter_names)
    baseline_values = np.array([float(getattr(base_gas, name)) for name in names])
    if np.any(baseline_values <= 0.0):
        raise ValueError("Calibrated coefficients must be positive")
    weights = (
        observations["weight"].to_numpy(dtype=float)
        if "weight" in observations
        else np.ones(len(observations))
    )
    clean_config = replace(config, measurement_noise_std=0.0)

    def gas_from_log_multipliers(values: np.ndarray) -> GasProperties:
        updates = {
            name: baseline * exp(multiplier)
            for name, baseline, multiplier in zip(
                names, baseline_values, values, strict=True
            )
        }
        return replace(base_gas, **updates)

    def residual(values: np.ndarray) -> np.ndarray:
        candidate = gas_from_log_multipliers(values)
        predictions = [
            float(
                simulate_plasma(
                    clean_config,
                    drive_scale=float(drive_scale),
                    custom_gas=candidate,
                ).summary["tail_sensor_mean"]
            )
            for drive_scale in observations["drive_scale"]
        ]
        return (
            np.asarray(predictions) - observations["sensor_observed"].to_numpy()
        ) * weights

    fit = least_squares(
        residual,
        np.zeros(len(names)),
        bounds=(-np.log(3.0), np.log(3.0)),
        max_nfev=maximum_function_evaluations,
    )
    calibrated = gas_from_log_multipliers(fit.x)
    predictions = [
        float(
            simulate_plasma(
                clean_config,
                drive_scale=float(drive_scale),
                custom_gas=calibrated,
            ).summary["tail_sensor_mean"]
        )
        for drive_scale in observations["drive_scale"]
    ]
    comparison = observations.copy()
    comparison["sensor_predicted"] = predictions
    comparison["residual"] = (
        comparison["sensor_predicted"] - comparison["sensor_observed"]
    )
    diagnostics: dict[str, float | int | str] = {
        "success": int(fit.success),
        "message": fit.message,
        "function_evaluations": int(fit.nfev),
        "cost": float(fit.cost),
        "rmse": float(
            np.sqrt(np.mean(np.square(comparison["residual"].to_numpy())))
        ),
    }
    for name in names:
        diagnostics[f"fitted_{name}"] = float(getattr(calibrated, name))
    return CalibrationResult(calibrated, comparison, diagnostics)


def _laplacian_neumann(values: np.ndarray, dx: float) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    return (padded[:-2] - 2.0 * padded[1:-1] + padded[2:]) / dx**2


def simulate_reaction_diffusion_1d(
    config: PlasmaConfig,
    spatial_config: SpatialModelConfig | None = None,
    custom_gas: GasProperties | None = None,
) -> SpatialSimulationResult:
    """Explicit finite-difference 1D reaction-diffusion demonstration."""

    validate_config(config)
    spatial = spatial_config or SpatialModelConfig()
    if spatial.grid_points < 5:
        raise ValueError("grid_points must be at least 5")
    if spatial.length_m <= 0.0 or spatial.duration_s <= 0.0 or spatial.dt_s <= 0.0:
        raise ValueError("Spatial length, duration, and dt must be positive")
    gas = resolve_gas(config, custom_gas)
    x_m = np.linspace(0.0, spatial.length_m, spatial.grid_points)
    dx = float(x_m[1] - x_m[0])
    maximum_diffusion = max(
        spatial.electron_diffusion_m2_s,
        spatial.positive_ion_diffusion_m2_s,
        spatial.negative_ion_diffusion_m2_s,
        spatial.metastable_diffusion_m2_s,
    )
    diffusion_number = maximum_diffusion * spatial.dt_s / dx**2
    if diffusion_number > 0.48:
        raise ValueError(
            f"Explicit diffusion step is unstable (D*dt/dx^2={diffusion_number:.3f})"
        )

    n_neutral = neutral_density_m3(config)
    density_scale = config.max_ionization_fraction * n_neutral
    initial = config.initial_ionization_fraction / max(
        config.max_ionization_fraction, 1.0e-30
    )
    perturbation = 1.0 + spatial.initial_perturbation_fraction * np.cos(
        2.0 * np.pi * x_m / spatial.length_m
    )
    electron = np.maximum(initial * perturbation, 0.0)
    positive = electron.copy()
    negative = np.zeros_like(electron)
    excitation = np.zeros_like(electron)
    metastable = np.zeros_like(electron)
    field_profile = spatial.field_edge_fraction + (
        1.0 - spatial.field_edge_fraction
    ) * np.sin(np.pi * x_m / spatial.length_m)
    base_field = blended_electric_field_v_m(config)
    period_s = 1.0 / config.pulse_frequency_hz
    flow_loss = config.flow_loss_per_sccm_s * max(config.gas_flow_sccm, 0.0)
    pressure_quench = 1.0 + gas.quenching_strength * (
        config.pressure_pa / STANDARD_PRESSURE_PA
    )
    deexcitation_rate = gas.deexcitation_rate_s * pressure_quench
    metastable_loss_rate = (
        gas.metastable_decay_rate_s
        + 90.0 * config.pressure_pa / STANDARD_PRESSURE_PA
    )

    total_steps = int(np.ceil(spatial.duration_s / spatial.dt_s))
    snapshot_stride = max(
        1, int(round(spatial.snapshot_interval_s / spatial.dt_s))
    )
    snapshots: list[dict[str, float]] = []
    previous_snapshot_electron = electron.copy()
    last_snapshot_change = float("nan")

    for step in range(total_steps + 1):
        time_s = min(step * spatial.dt_s, spatial.duration_s)
        pulse_on = float((time_s % period_s) < config.pulse_width_s)
        field_fraction = (
            pulse_on + config.afterglow_field_fraction * (1.0 - pulse_on)
        )
        local_field = base_field * field_profile * field_fraction
        local_td = local_field / n_neutral / 1.0e-21
        activation = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    (local_td - gas.ionization_threshold_td)
                    / gas.transition_width_td,
                    -60.0,
                    60.0,
                )
            )
        )

        if step % snapshot_stride == 0 or step == total_steps:
            emission = (
                gas.radiative_fraction
                * deexcitation_rate
                * excitation
                / 1_000.0
                + 0.08 * gas.radiative_fraction * metastable
            )
            for index, position in enumerate(x_m):
                snapshots.append(
                    {
                        "time_s": time_s,
                        "x_m": position,
                        "field_profile": field_profile[index],
                        "reduced_field_td": local_td[index],
                        "electron_density_m3": electron[index] * density_scale,
                        "positive_ion_density_m3": positive[index] * density_scale,
                        "negative_ion_density_m3": negative[index] * density_scale,
                        "excitation_fraction": excitation[index],
                        "metastable_fraction": metastable[index],
                        "emission_intensity_au": emission[index],
                    }
                )
            last_snapshot_change = float(
                np.linalg.norm(electron - previous_snapshot_electron)
                / (np.linalg.norm(previous_snapshot_electron) + 1.0e-12)
            )
            previous_snapshot_electron = electron.copy()
        if step == total_steps:
            break

        capacity = np.maximum(
            0.0,
            1.0
            - (electron + negative) / config.density_ceiling_normalized,
        )
        population = electron + config.seed_floor_normalized
        ionization = (
            gas.max_ionization_rate_s * activation * population * capacity
        )
        stepwise = (
            gas.stepwise_ionization_rate_s
            * metastable
            * population
            * capacity
        )
        attachment = (
            gas.electron_attachment_rate_s
            * (1.0 - 0.65 * activation)
            * electron
        )
        detachment = gas.electron_detachment_rate_s * activation * negative
        recombination = gas.recombination_rate_s * electron * positive
        ion_ion = gas.ion_ion_recombination_rate_s * positive * negative
        excitation_drive = (
            gas.excitation_rate_s
            * activation
            * electron
            / (0.12 + electron)
            * (1.0 - excitation)
        )
        metastable_drive = (
            gas.metastable_production_rate_s
            * activation
            * electron
            / (0.10 + electron)
            * (1.0 - metastable)
        )

        electron_rate = (
            ionization
            + stepwise
            + detachment
            - attachment
            - recombination
            - (config.electron_wall_loss_rate_s + flow_loss) * electron
            + spatial.electron_diffusion_m2_s
            * _laplacian_neumann(electron, dx)
        )
        positive_rate = (
            ionization
            + stepwise
            - recombination
            - ion_ion
            - (config.ion_wall_loss_rate_s + 0.35 * flow_loss) * positive
            + spatial.positive_ion_diffusion_m2_s
            * _laplacian_neumann(positive, dx)
        )
        negative_rate = (
            attachment
            - detachment
            - ion_ion
            - (65.0 + 0.25 * flow_loss) * negative
            + spatial.negative_ion_diffusion_m2_s
            * _laplacian_neumann(negative, dx)
        )
        excitation_rate = excitation_drive - deexcitation_rate * excitation
        metastable_rate = (
            metastable_drive
            - metastable_loss_rate * metastable
            - stepwise
            + spatial.metastable_diffusion_m2_s
            * _laplacian_neumann(metastable, dx)
        )

        electron = np.clip(
            electron + spatial.dt_s * electron_rate,
            0.0,
            config.density_ceiling_normalized,
        )
        positive = np.clip(
            positive + spatial.dt_s * positive_rate,
            0.0,
            config.density_ceiling_normalized,
        )
        negative = np.clip(
            negative + spatial.dt_s * negative_rate,
            0.0,
            config.density_ceiling_normalized,
        )
        excitation = np.clip(
            excitation + spatial.dt_s * excitation_rate, 0.0, 1.0
        )
        metastable = np.clip(
            metastable + spatial.dt_s * metastable_rate, 0.0, 1.0
        )

    frame = pd.DataFrame(snapshots)
    final = frame[frame["time_s"] == frame["time_s"].max()]
    final_electron = final["electron_density_m3"].to_numpy()
    center_index = len(final) // 2
    edge_mean = 0.5 * (final_electron[0] + final_electron[-1])
    spatial_cv = float(
        np.std(final_electron)
        / (abs(float(np.mean(final_electron))) + 1.0e-30)
    )
    summary: dict[str, float | int | str] = {
        "gas": gas.name,
        "grid_points": spatial.grid_points,
        "time_steps": total_steps,
        "diffusion_number": diffusion_number,
        "peak_electron_density_m3": float(final_electron.max()),
        "peak_position_m": float(
            final.iloc[int(np.argmax(final_electron))]["x_m"]
        ),
        "center_to_edge_density_ratio": float(
            final_electron[center_index] / max(edge_mean, 1.0e-30)
        ),
        "spatial_coefficient_of_variation": spatial_cv,
        "spatial_uniformity_metric": float(exp(-spatial_cv)),
        "last_snapshot_relative_change": last_snapshot_change,
    }
    return SpatialSimulationResult(frame, summary, config, spatial, gas)
