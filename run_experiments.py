"""Run three scenarios, logic tests, sweeps, CSV export, and plotting."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_models import (
    AdvancedModelOptions,
    SpatialModelConfig,
    calibrate_gas_to_observations,
    monte_carlo_uncertainty,
    simulate_advanced_plasma,
    simulate_reaction_diffusion_1d,
)
from plasma_model import (
    GAS_LIBRARY,
    PlasmaConfig,
    full_adder_truth_table,
    logic_truth_tables,
    parameter_sensitivity,
    schmitt_trigger,
    simulate_plasma,
    sr_latch_sequence,
    threshold_sweep,
)


def scenario_configs() -> dict[str, PlasmaConfig]:
    common = PlasmaConfig(duration_s=0.030, dt_s=2.0e-5, random_seed=2026)
    return {
        "S1_subthreshold_argon": replace(
            common,
            gas="argon",
            pressure_pa=4_000.0,
            temperature_k=300.0,
            electric_field_v_m=80_000.0,
            applied_voltage_v=400.0,
            pulse_frequency_hz=2_000.0,
            pulse_width_s=1.8e-4,
            gas_flow_sccm=60.0,
            measurement_noise_std=0.020,
        ),
        "S2_stable_neon": replace(
            common,
            gas="neon",
            pressure_pa=3_500.0,
            temperature_k=310.0,
            electric_field_v_m=220_000.0,
            applied_voltage_v=1_100.0,
            pulse_frequency_hz=2_000.0,
            pulse_width_s=2.2e-4,
            gas_flow_sccm=40.0,
            measurement_noise_std=0.020,
        ),
        "S3_noisy_air_threshold": replace(
            common,
            gas="air",
            pressure_pa=5_000.0,
            temperature_k=320.0,
            electric_field_v_m=220_000.0,
            applied_voltage_v=1_100.0,
            pulse_frequency_hz=1_500.0,
            pulse_width_s=4.0e-4,
            gas_flow_sccm=90.0,
            measurement_noise_std=0.085,
        ),
    }


def logic_config() -> PlasmaConfig:
    return PlasmaConfig(
        gas="argon",
        pressure_pa=4_000.0,
        temperature_k=300.0,
        electric_field_v_m=175_000.0,
        applied_voltage_v=875.0,
        pulse_frequency_hz=2_000.0,
        pulse_width_s=2.0e-4,
        gas_flow_sccm=45.0,
        measurement_noise_std=0.025,
        duration_s=0.035,
        binary_threshold=0.45,
        random_seed=2026,
    )


def save_scenario_plot(name: str, data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    t_ms = 1_000.0 * data["time_s"]
    axes[0, 0].plot(t_ms, data["ionization_fraction"], color="#6f42c1")
    axes[0, 0].set_ylabel("Ionization fraction")
    axes[0, 0].set_title("Ionization")

    axes[0, 1].plot(t_ms, data["electron_density_m3"], color="#0068b5")
    axes[0, 1].set_ylabel("Electron density (m$^{-3}$)")
    axes[0, 1].set_title("Electron density")

    axes[1, 0].plot(t_ms, data["emission_intensity_au"], color="#d97706")
    axes[1, 0].set_ylabel("Emission intensity (a.u.)")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_title("Optical emission")

    axes[1, 1].plot(
        t_ms, data["conductivity_s_m"], color="#18864b", label="Conductivity"
    )
    sensor_axis = axes[1, 1].twinx()
    sensor_axis.plot(
        t_ms,
        data["sensor_measured"],
        color="#b42318",
        alpha=0.55,
        linewidth=0.9,
        label="Measured sensor",
    )
    axes[1, 1].set_ylabel("Conductivity (S/m)")
    sensor_axis.set_ylabel("Sensor response")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].set_title("Conductivity and detector")

    fig.suptitle(name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160)
    plt.close(fig)


def save_comparison_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["scenario"].str.replace("_", " ", regex=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].bar(labels, summary["tail_sensor_mean"], color="#337ab7")
    axes[0].set_ylabel("Tail sensor mean")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Detected response")
    axes[1].bar(labels, summary["stability_metric"], color="#18864b")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Stability metric (0-1)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Cycle stability")
    axes[2].bar(labels, summary["noise_robustness_metric"], color="#d97706")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Noise robustness (0-1)")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_title("Threshold agreement")
    fig.tight_layout()
    fig.savefig(output_dir / "scenario_comparison.png", dpi=160)
    plt.close(fig)


def save_analysis_plots(
    threshold_data: pd.DataFrame,
    sensitivity: pd.DataFrame,
    field_pressure: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        threshold_data["threshold"],
        threshold_data["tail_high_fraction"],
        marker="o",
    )
    ax.set_xlabel("Binary threshold")
    ax.set_ylabel("Tail high fraction")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Threshold sweep")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_sweep.png", dpi=160)
    plt.close(fig)

    valid = sensitivity.dropna(subset=["absolute_sensitivity"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(valid["normalized_sensitivity"] >= 0, "#337ab7", "#b42318")
    ax.barh(valid["parameter"], valid["normalized_sensitivity"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized sensitivity")
    ax.set_title("One-at-a-time sensitivity (+/-10%)")
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_sensitivity.png", dpi=160)
    plt.close(fig)

    pivot = field_pressure.pivot(
        index="pressure_pa", columns="electric_field_v_m", values="tail_sensor_mean"
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min() / 1_000.0,
            pivot.columns.max() / 1_000.0,
            pivot.index.min(),
            pivot.index.max(),
        ],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Electric field (kV/m)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Field-pressure parameter sweep")
    fig.colorbar(image, ax=ax, label="Tail sensor mean")
    fig.tight_layout()
    fig.savefig(output_dir / "field_pressure_sweep.png", dpi=160)
    plt.close(fig)


def save_logic_plot(calibration: dict[str, float], output_dir: Path) -> None:
    active_counts = np.array([0, 1, 2])
    responses = np.array(
        [
            calibration["response_0_active"],
            calibration["response_1_active"],
            calibration["response_2_active"],
        ]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(active_counts, responses, color=["#6b7280", "#337ab7", "#18864b"])
    ax.axhline(
        calibration["or_threshold"],
        color="#d97706",
        linestyle="--",
        label="OR / XOR lower threshold",
    )
    ax.axhline(
        calibration["and_threshold"],
        color="#b42318",
        linestyle="--",
        label="AND / XOR upper threshold",
    )
    ax.set_xticks(active_counts)
    ax.set_xlabel("Number of active physical inputs")
    ax.set_ylabel("Calibrated sensor response")
    ax.set_ylim(0.0, min(1.0, max(responses) * 1.25))
    ax.set_title("Physical response levels used for threshold logic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "logic_response_levels.png", dpi=160)
    plt.close(fig)


def save_advanced_plot(name: str, data: pd.DataFrame, output_dir: Path) -> None:
    time_ms = 1_000.0 * data["time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    axes[0, 0].plot(
        time_ms, data["electron_density_m3"], label="Electrons", color="#0068b5"
    )
    axes[0, 0].plot(
        time_ms,
        data["negative_ion_density_m3"],
        label="Negative ions",
        color="#b42318",
    )
    axes[0, 0].set_ylabel("Density (m$^{-3}$)")
    axes[0, 0].set_title("Charged species")
    axes[0, 0].legend()

    axes[0, 1].plot(
        time_ms,
        data["excitation_fraction"],
        label="Excitation",
        color="#d97706",
    )
    axes[0, 1].plot(
        time_ms,
        data["metastable_fraction"],
        label="Metastable",
        color="#6f42c1",
    )
    axes[0, 1].set_ylabel("Normalized population")
    axes[0, 1].set_title("Excited populations")
    axes[0, 1].legend()

    axes[1, 0].plot(
        time_ms, data["electron_energy_ev"], color="#18864b"
    )
    axes[1, 0].set_ylabel("Mean electron energy (eV)")
    axes[1, 0].set_xlabel("Time (ms)")
    axes[1, 0].set_title("Two-temperature proxy")

    axes[1, 1].plot(
        time_ms, data["sensor_clean"], color="#1f2937", label="Clean sensor"
    )
    axes[1, 1].plot(
        time_ms,
        data["sensor_measured"],
        color="#b42318",
        alpha=0.45,
        linewidth=0.8,
        label="Measured sensor",
    )
    axes[1, 1].set_ylabel("Sensor response")
    axes[1, 1].set_xlabel("Time (ms)")
    axes[1, 1].set_title("Detector response")
    axes[1, 1].legend()
    fig.suptitle(name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=160)
    plt.close(fig)


def save_monte_carlo_plot(samples: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(
        samples["tail_sensor_mean"], bins=14, color="#337ab7", edgecolor="white"
    )
    axes[0].axvline(0.45, color="#b42318", linestyle="--", label="Threshold")
    axes[0].set_xlabel("Tail sensor mean")
    axes[0].set_ylabel("Sample count")
    axes[0].set_title("Monte Carlo response distribution")
    axes[0].legend()
    axes[1].scatter(
        samples["pressure_pa"],
        samples["tail_sensor_mean"],
        c=samples["pulse_width_s"],
        cmap="viridis",
        s=28,
    )
    axes[1].set_xlabel("Pressure (Pa)")
    axes[1].set_ylabel("Tail sensor mean")
    axes[1].set_title("Pressure and pulse-width uncertainty")
    fig.tight_layout()
    fig.savefig(output_dir / "monte_carlo_uncertainty.png", dpi=160)
    plt.close(fig)


def save_calibration_plot(comparison: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        comparison["drive_scale"],
        comparison["sensor_observed"],
        marker="o",
        label="Synthetic observations",
    )
    ax.plot(
        comparison["drive_scale"],
        comparison["sensor_predicted"],
        marker="s",
        linestyle="--",
        label="Fitted model",
    )
    ax.set_xlabel("Drive scale")
    ax.set_ylabel("Tail sensor response")
    ax.set_title("CSV-ready coefficient calibration")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "coefficient_calibration.png", dpi=160)
    plt.close(fig)


def save_stateful_logic_plot(
    hysteresis: pd.DataFrame, latch: pd.DataFrame, output_dir: Path
) -> None:
    tail = hysteresis[
        hysteresis["time_s"] >= hysteresis["time_s"].max() - 0.010
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2))
    axes[0].plot(
        1_000.0 * tail["time_s"],
        tail["sensor_measured"],
        color="#6b7280",
        linewidth=0.8,
        label="Measured sensor",
    )
    axes[0].axhline(0.45, color="#337ab7", linestyle=":", label="Single threshold")
    axes[0].axhline(0.40, color="#d97706", linestyle="--", label="Schmitt lower")
    axes[0].axhline(0.50, color="#b42318", linestyle="--", label="Schmitt upper")
    axes[0].set_ylabel("Sensor response")
    axes[0].set_title("Near-threshold sensor, final 10 ms")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].step(
        1_000.0 * tail["time_s"],
        tail["simple_binary"],
        where="post",
        label="Single threshold",
    )
    axes[1].step(
        1_000.0 * tail["time_s"],
        tail["schmitt_binary"],
        where="post",
        label="Schmitt output",
    )
    axes[1].set_ylabel("Digital state")
    axes[1].set_yticks([0, 1])
    axes[1].set_title("Hysteresis suppresses threshold chatter")
    axes[1].legend(loc="upper right")

    axes[2].step(latch["step"], latch["SET"], where="post", label="SET")
    axes[2].step(latch["step"], latch["RESET"], where="post", label="RESET")
    axes[2].step(
        latch["step"], latch["Q"], where="post", linewidth=2.0, label="Q"
    )
    axes[2].set_xlabel("Sequence step")
    axes[2].set_ylabel("Logic state")
    axes[2].set_yticks([0, 1])
    axes[2].set_title("Conceptual SR latch memory")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stateful_logic.png", dpi=160)
    plt.close(fig)


def save_spatial_plot(data: pd.DataFrame, output_dir: Path) -> None:
    electron_pivot = data.pivot(
        index="time_s", columns="x_m", values="electron_density_m3"
    )
    final_time = data["time_s"].max()
    final = data[data["time_s"] == final_time]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    image = axes[0].imshow(
        electron_pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[
            1_000.0 * electron_pivot.columns.min(),
            1_000.0 * electron_pivot.columns.max(),
            1_000.0 * electron_pivot.index.min(),
            1_000.0 * electron_pivot.index.max(),
        ],
        cmap="magma",
    )
    axes[0].set_xlabel("Position (mm)")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("1D electron-density evolution")
    fig.colorbar(image, ax=axes[0], label="Electron density (m$^{-3}$)")
    axes[1].plot(
        1_000.0 * final["x_m"],
        final["electron_density_m3"],
        label="Electrons",
    )
    axes[1].plot(
        1_000.0 * final["x_m"],
        final["negative_ion_density_m3"],
        label="Negative ions",
    )
    axes[1].set_xlabel("Position (mm)")
    axes[1].set_ylabel("Density (m$^{-3}$)")
    axes[1].set_title("Final spatial profiles")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reaction_diffusion_1d.png", dpi=160)
    plt.close(fig)


def field_pressure_sweep(config: PlasmaConfig) -> pd.DataFrame:
    fields = np.linspace(100_000.0, 240_000.0, 8)
    pressures = np.linspace(3_000.0, 6_000.0, 7)
    rows = []
    for pressure in pressures:
        for field in fields:
            swept = replace(
                config,
                pressure_pa=float(pressure),
                electric_field_v_m=float(field),
                applied_voltage_v=float(field * config.electrode_gap_m),
                measurement_noise_std=0.0,
            )
            result = simulate_plasma(swept)
            rows.append(
                {
                    "pressure_pa": pressure,
                    "electric_field_v_m": field,
                    "base_reduced_field_td": result.summary["base_reduced_field_td"],
                    "tail_sensor_mean": result.summary["tail_sensor_mean"],
                    "binary_output": result.summary["tail_binary_output"],
                    "stability_metric": result.summary["stability_metric"],
                }
            )
    return pd.DataFrame(rows)


def write_text_summary(
    scenario_summary: pd.DataFrame,
    calibration: dict[str, float],
    sensitivity: pd.DataFrame,
    advanced_summary: dict[str, float | int | str],
    monte_carlo_summary: pd.DataFrame,
    calibration_diagnostics: dict[str, float | int | str],
    spatial_summary: dict[str, float | int | str],
    stateful_metrics: dict[str, int],
    output_dir: Path,
) -> None:
    most_sensitive = sensitivity.dropna(subset=["absolute_sensitivity"]).iloc[0]
    lines = [
        "IONIZATION-BASED LOGIC SIMULATION SUMMARY",
        "",
        "Scenario comparison:",
        scenario_summary.to_string(index=False),
        "",
        "Logic calibration:",
        *[f"{key}: {value:.6g}" for key, value in calibration.items()],
        "",
        "Advanced air model:",
        *[f"{key}: {value}" for key, value in advanced_summary.items()],
        "",
        "Monte Carlo uncertainty:",
        monte_carlo_summary.to_string(index=False),
        "",
        "Coefficient calibration:",
        *[f"{key}: {value}" for key, value in calibration_diagnostics.items()],
        "",
        "1D reaction-diffusion:",
        *[f"{key}: {value}" for key, value in spatial_summary.items()],
        "",
        "Stateful logic:",
        *[f"{key}: {value}" for key, value in stateful_metrics.items()],
        "",
        (
            "Most sensitive parameter: "
            f"{most_sensitive['parameter']} "
            f"(normalized sensitivity={most_sensitive['normalized_sensitivity']:.4f})"
        ),
        "",
        "Interpretation: output is a conceptual mean-field detector response, "
        "not a laboratory prediction or control recommendation.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def run_all(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    scenario_results = {}
    for name, config in scenario_configs().items():
        result = simulate_plasma(config)
        scenario_results[name] = result
        result.data.to_csv(output_dir / f"{name}.csv", index=False)
        save_scenario_plot(name, result.data, output_dir)
        summaries.append({"scenario": name, **result.summary})

    scenario_summary = pd.DataFrame(summaries)
    scenario_summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    save_comparison_plot(scenario_summary, output_dir)

    logic = logic_config()
    logic_table, not_table, calibration = logic_truth_tables(logic)
    full_adder, full_adder_calibration = full_adder_truth_table(logic)
    logic_table.to_csv(output_dir / "logic_truth_table.csv", index=False)
    not_table.to_csv(output_dir / "not_truth_table.csv", index=False)
    full_adder.to_csv(output_dir / "full_adder_truth_table.csv", index=False)
    pd.DataFrame([calibration]).to_csv(
        output_dir / "logic_calibration.csv", index=False
    )
    pd.DataFrame([full_adder_calibration]).to_csv(
        output_dir / "full_adder_calibration.csv", index=False
    )
    save_logic_plot(calibration, output_dir)

    base_result = simulate_plasma(logic)
    thresholds = threshold_sweep(base_result, np.linspace(0.20, 0.80, 13))
    thresholds.to_csv(output_dir / "threshold_sweep.csv", index=False)
    sensitivity = parameter_sensitivity(logic, relative_step=0.10)
    sensitivity.to_csv(output_dir / "parameter_sensitivity.csv", index=False)
    field_pressure = field_pressure_sweep(logic)
    field_pressure.to_csv(output_dir / "field_pressure_sweep.csv", index=False)
    save_analysis_plots(thresholds, sensitivity, field_pressure, output_dir)

    advanced_air_config = scenario_configs()["S3_noisy_air_threshold"]
    advanced_air = simulate_advanced_plasma(advanced_air_config)
    advanced_air.data.to_csv(output_dir / "advanced_air_bdf.csv", index=False)
    pd.DataFrame([advanced_air.summary]).to_csv(
        output_dir / "advanced_air_summary.csv", index=False
    )
    save_advanced_plot("advanced_air_bdf", advanced_air.data, output_dir)

    solver_config = replace(
        scenario_configs()["S2_stable_neon"],
        duration_s=0.012,
        measurement_noise_std=0.0,
    )
    solver_rows = []
    for solver_method in ("BDF", "RK45"):
        solver_result = simulate_advanced_plasma(
            solver_config,
            options=AdvancedModelOptions(solver_method=solver_method),
        )
        solver_rows.append(
            {
                "solver": solver_method,
                "tail_sensor_mean": solver_result.summary["tail_sensor_mean"],
                "peak_electron_density_m3": solver_result.summary[
                    "peak_electron_density_m3"
                ],
                "function_evaluations": solver_result.summary[
                    "solver_function_evaluations"
                ],
                "stability_metric": solver_result.summary["stability_metric"],
            }
        )
    solver_comparison = pd.DataFrame(solver_rows)
    solver_comparison["relative_sensor_difference_from_bdf"] = (
        solver_comparison["tail_sensor_mean"]
        - float(solver_comparison.iloc[0]["tail_sensor_mean"])
    ) / max(abs(float(solver_comparison.iloc[0]["tail_sensor_mean"])), 1.0e-12)
    solver_comparison.to_csv(
        output_dir / "adaptive_solver_comparison.csv", index=False
    )

    monte_carlo_samples, monte_carlo_summary = monte_carlo_uncertainty(
        advanced_air_config,
        sample_count=60,
        relative_standard_deviation=0.08,
        random_seed=2026,
    )
    monte_carlo_samples.to_csv(
        output_dir / "monte_carlo_samples.csv", index=False
    )
    monte_carlo_summary.to_csv(
        output_dir / "monte_carlo_summary.csv", index=False
    )
    save_monte_carlo_plot(monte_carlo_samples, output_dir)

    calibration_config = replace(
        logic,
        duration_s=0.018,
        dt_s=4.0e-5,
        measurement_noise_std=0.0,
    )
    target_gas = replace(
        GAS_LIBRARY["argon"],
        ionization_threshold_td=122.0,
        max_ionization_rate_s=4_050.0,
        recombination_rate_s=420.0,
    )
    calibration_drives = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
    deterministic_offsets = np.array([0.0010, -0.0015, 0.0012, -0.0008, 0.0005])
    observed_values = []
    for drive, offset in zip(
        calibration_drives, deterministic_offsets, strict=True
    ):
        predicted = float(
            simulate_plasma(
                calibration_config,
                drive_scale=float(drive),
                custom_gas=target_gas,
            ).summary["tail_sensor_mean"]
        )
        observed_values.append(float(np.clip(predicted + offset, 0.0, 1.0)))
    observations = pd.DataFrame(
        {
            "drive_scale": calibration_drives,
            "sensor_observed": observed_values,
            "weight": np.ones(len(calibration_drives)),
        }
    )
    observations.to_csv(
        output_dir / "calibration_observations.csv", index=False
    )
    fitted = calibrate_gas_to_observations(
        calibration_config,
        GAS_LIBRARY["argon"],
        observations,
        maximum_function_evaluations=35,
    )
    fitted.comparison.to_csv(
        output_dir / "calibration_comparison.csv", index=False
    )
    pd.DataFrame([fitted.diagnostics]).to_csv(
        output_dir / "calibration_diagnostics.csv", index=False
    )
    save_calibration_plot(fitted.comparison, output_dir)

    noisy_result = scenario_results["S3_noisy_air_threshold"]
    measured = noisy_result.data["sensor_measured"].to_numpy()
    simple_binary = (measured >= noisy_result.config.binary_threshold).astype(int)
    schmitt_binary = schmitt_trigger(
        measured, lower_threshold=0.40, upper_threshold=0.50
    )
    hysteresis = pd.DataFrame(
        {
            "time_s": noisy_result.data["time_s"],
            "sensor_measured": measured,
            "simple_binary": simple_binary,
            "schmitt_binary": schmitt_binary,
        }
    )
    hysteresis.to_csv(output_dir / "schmitt_hysteresis.csv", index=False)
    latch = sr_latch_sequence(
        set_input=[0, 1, 0, 0, 0, 0, 1, 0],
        reset_input=[0, 0, 0, 1, 0, 0, 1, 0],
    )
    latch.to_csv(output_dir / "sr_latch_sequence.csv", index=False)
    simple_transitions = int(np.count_nonzero(np.diff(simple_binary)))
    schmitt_transitions = int(np.count_nonzero(np.diff(schmitt_binary)))
    stateful_metrics = {
        "simple_threshold_transitions": simple_transitions,
        "schmitt_transitions": schmitt_transitions,
        "transitions_suppressed": simple_transitions - schmitt_transitions,
        "latch_invalid_steps": int(latch["invalid_input"].sum()),
    }
    pd.DataFrame([stateful_metrics]).to_csv(
        output_dir / "stateful_logic_metrics.csv", index=False
    )
    save_stateful_logic_plot(hysteresis, latch, output_dir)

    spatial_result = simulate_reaction_diffusion_1d(
        replace(advanced_air_config, measurement_noise_std=0.0),
        SpatialModelConfig(
            duration_s=0.020,
            snapshot_interval_s=1.0 / advanced_air_config.pulse_frequency_hz,
        ),
    )
    spatial_result.data.to_csv(
        output_dir / "reaction_diffusion_1d.csv", index=False
    )
    pd.DataFrame([spatial_result.summary]).to_csv(
        output_dir / "reaction_diffusion_1d_summary.csv", index=False
    )
    save_spatial_plot(spatial_result.data, output_dir)

    write_text_summary(
        scenario_summary,
        calibration,
        sensitivity,
        advanced_air.summary,
        monte_carlo_summary,
        fitted.diagnostics,
        spatial_result.summary,
        stateful_metrics,
        output_dir,
    )

    print("\nSCENARIO RESULTS")
    print(
        scenario_summary[
            [
                "scenario",
                "tail_sensor_mean",
                "tail_binary_output",
                "delay_s",
                "stability_metric",
                "noise_robustness_metric",
                "final_regime",
            ]
        ].to_string(index=False)
    )
    print("\nLOGIC TRUTH TABLE")
    print(logic_table.to_string(index=False))
    print("\nNOT TRUTH TABLE")
    print(not_table.to_string(index=False))
    print("\nFULL ADDER TRUTH TABLE")
    print(full_adder.to_string(index=False))
    print("\nADAPTIVE SOLVER COMPARISON")
    print(solver_comparison.to_string(index=False))
    print("\nMONTE CARLO SUMMARY")
    print(monte_carlo_summary.to_string(index=False))
    print("\nCALIBRATION DIAGNOSTICS")
    print(pd.DataFrame([fitted.diagnostics]).to_string(index=False))
    print("\nSTATEFUL LOGIC METRICS")
    print(pd.DataFrame([stateful_metrics]).to_string(index=False))
    print("\n1D SPATIAL SUMMARY")
    print(pd.DataFrame([spatial_result.summary]).to_string(index=False))
    print("\nTOP SENSITIVITIES")
    print(
        sensitivity[
            ["parameter", "normalized_sensitivity", "absolute_sensitivity"]
        ].head(5).to_string(index=False)
    )
    print(f"\nFiles written to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the educational ionization-based logic simulation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV, PNG, and text outputs.",
    )
    args = parser.parse_args()
    run_all(args.output_dir)


if __name__ == "__main__":
    main()
