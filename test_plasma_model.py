"""Focused regression tests for the educational plasma model."""

from __future__ import annotations

import unittest
from dataclasses import replace

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
    GasProperties,
    PlasmaConfig,
    full_adder_truth_table,
    logistic_activation,
    logic_truth_tables,
    schmitt_trigger,
    simulate_plasma,
    sr_latch_sequence,
    threshold_sweep,
)


class PlasmaModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PlasmaConfig(
            duration_s=0.020,
            dt_s=2.0e-5,
            measurement_noise_std=0.0,
            random_seed=42,
        )

    def test_states_are_finite_and_nonnegative(self) -> None:
        result = simulate_plasma(self.config)
        columns = [
            "ionization_fraction",
            "electron_density_m3",
            "ion_density_m3",
            "excitation_fraction",
            "conductivity_s_m",
            "emission_intensity_au",
        ]
        values = result.data[columns].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0.0).all())

    def test_higher_drive_gives_stronger_response(self) -> None:
        low = simulate_plasma(self.config, drive_scale=0.35)
        high = simulate_plasma(self.config, drive_scale=1.0)
        self.assertGreater(
            float(high.summary["tail_sensor_mean"]),
            float(low.summary["tail_sensor_mean"]),
        )

    def _logic_tables(self):
        logic_config = replace(
            self.config,
            duration_s=0.035,
            electric_field_v_m=175_000.0,
            applied_voltage_v=875.0,
            measurement_noise_std=0.02,
        )
        return logic_truth_tables(logic_config)

    def test_and_gate(self) -> None:
        table, _, _ = self._logic_tables()
        self.assertEqual(table["AND"].tolist(), [0, 0, 0, 1])

    def test_or_gate(self) -> None:
        table, _, _ = self._logic_tables()
        self.assertEqual(table["OR"].tolist(), [0, 1, 1, 1])

    def test_xor_gate(self) -> None:
        table, _, _ = self._logic_tables()
        self.assertEqual(table["XOR"].tolist(), [0, 1, 1, 0])

    def test_half_adder_and_not(self) -> None:
        table, not_table, _ = self._logic_tables()
        self.assertEqual(table["HALF_ADDER_SUM"].tolist(), [0, 1, 1, 0])
        self.assertEqual(table["HALF_ADDER_CARRY"].tolist(), [0, 0, 0, 1])
        self.assertEqual(not_table["NOT"].tolist(), [1, 0])

    def test_custom_gas_is_supported(self) -> None:
        custom = GasProperties(
            name="custom_demo",
            ionization_threshold_td=125.0,
            transition_width_td=16.0,
            max_ionization_rate_s=4_000.0,
            recombination_rate_s=400.0,
            excitation_rate_s=3_000.0,
            deexcitation_rate_s=2_000.0,
            radiative_fraction=0.6,
            electron_mobility_ref_m2_v_s=0.09,
        )
        config = replace(self.config, gas="custom_demo")
        result = simulate_plasma(config, custom_gas=custom)
        self.assertEqual(result.summary["gas"], "custom_demo")

    def test_threshold_sweep_is_monotonic(self) -> None:
        result = simulate_plasma(self.config)
        sweep = threshold_sweep(result, [0.2, 0.4, 0.6, 0.8])
        high_fraction = sweep["tail_high_fraction"].to_numpy()
        self.assertTrue(np.all(np.diff(high_fraction) <= 1.0e-12))

    def test_ionization_activation_is_smooth(self) -> None:
        threshold = 115.0
        width = 14.0
        below = logistic_activation(threshold - 0.01, threshold, width)
        midpoint = logistic_activation(threshold, threshold, width)
        above = logistic_activation(threshold + 0.01, threshold, width)
        self.assertLess(below, midpoint)
        self.assertAlmostEqual(midpoint, 0.5)
        self.assertGreater(above, midpoint)
        self.assertLess(above - below, 0.01)

    def test_plasma_state_persists_after_pulse(self) -> None:
        config = PlasmaConfig(
            gas="argon",
            pressure_pa=4_000.0,
            temperature_k=300.0,
            electric_field_v_m=220_000.0,
            applied_voltage_v=1_100.0,
            pulse_frequency_hz=500.0,
            pulse_width_s=4.0e-4,
            gas_flow_sccm=40.0,
            measurement_noise_std=0.0,
            initial_ionization_fraction=1.0e-10,
            duration_s=1.5e-3,
            dt_s=1.0e-5,
        )
        result = simulate_plasma(config)
        data = result.data
        pulse_end_index = int(round(config.pulse_width_s / config.dt_s))
        afterglow_index = pulse_end_index + int(round(2.0e-4 / config.dt_s))
        density_at_pulse_end = float(
            data.iloc[pulse_end_index]["electron_density_m3"]
        )
        density_in_afterglow = float(
            data.iloc[afterglow_index]["electron_density_m3"]
        )
        initial_density = float(data.iloc[0]["electron_density_m3"])
        self.assertGreater(density_in_afterglow, initial_density)
        self.assertGreater(density_at_pulse_end, density_in_afterglow)

    def test_full_adder(self) -> None:
        config = replace(
            self.config,
            duration_s=0.025,
            electric_field_v_m=175_000.0,
            applied_voltage_v=875.0,
        )
        table, _ = full_adder_truth_table(config)
        self.assertEqual(table["SUM"].tolist(), [0, 1, 1, 0, 1, 0, 0, 1])
        self.assertEqual(
            table["CARRY_OUT"].tolist(), [0, 0, 0, 1, 0, 1, 1, 1]
        )

    def test_schmitt_trigger_reduces_chatter(self) -> None:
        signal = np.array([0.30, 0.52, 0.47, 0.44, 0.48, 0.39, 0.46, 0.38])
        simple = (signal >= 0.45).astype(int)
        schmitt = schmitt_trigger(signal, 0.40, 0.50)
        self.assertLess(
            np.count_nonzero(np.diff(schmitt)),
            np.count_nonzero(np.diff(simple)),
        )

    def test_sr_latch_retains_state(self) -> None:
        table = sr_latch_sequence([0, 1, 0, 0], [0, 0, 0, 1])
        self.assertEqual(table["Q"].tolist(), [0, 1, 1, 0])

    def test_advanced_species_are_nonnegative(self) -> None:
        config = replace(
            self.config,
            gas="air",
            duration_s=0.004,
            electric_field_v_m=220_000.0,
            applied_voltage_v=1_100.0,
            pulse_width_s=2.0e-4,
        )
        result = simulate_advanced_plasma(config)
        columns = [
            "electron_density_m3",
            "negative_ion_density_m3",
            "metastable_fraction",
            "electron_energy_ev",
        ]
        self.assertTrue(np.isfinite(result.data[columns].to_numpy()).all())
        self.assertTrue((result.data[columns].to_numpy() >= 0.0).all())
        self.assertGreater(result.data["negative_ion_density_m3"].max(), 0.0)
        self.assertGreater(result.data["metastable_fraction"].max(), 0.0)

    def test_adaptive_solvers_agree(self) -> None:
        config = replace(
            self.config,
            duration_s=0.003,
            measurement_noise_std=0.0,
        )
        bdf = simulate_advanced_plasma(
            config, options=AdvancedModelOptions(solver_method="BDF")
        )
        rk45 = simulate_advanced_plasma(
            config, options=AdvancedModelOptions(solver_method="RK45")
        )
        self.assertAlmostEqual(
            float(bdf.summary["tail_sensor_mean"]),
            float(rk45.summary["tail_sensor_mean"]),
            delta=2.0e-3,
        )

    def test_monte_carlo_summary(self) -> None:
        samples, summary = monte_carlo_uncertainty(
            replace(self.config, duration_s=0.006),
            sample_count=6,
            random_seed=11,
        )
        self.assertEqual(len(samples), 6)
        self.assertTrue(0.0 <= summary.iloc[0]["probability_output_1"] <= 1.0)
        self.assertLessEqual(
            summary.iloc[0]["sensor_q025"], summary.iloc[0]["sensor_q975"]
        )

    def test_coefficient_calibration(self) -> None:
        config = replace(
            self.config,
            duration_s=0.012,
            dt_s=4.0e-5,
            measurement_noise_std=0.0,
        )
        target = replace(
            GAS_LIBRARY["argon"], ionization_threshold_td=125.0
        )
        drives = [0.55, 0.70, 0.85]
        observations = pd.DataFrame(
            {
                "drive_scale": drives,
                "sensor_observed": [
                    simulate_plasma(
                        config, drive_scale=drive, custom_gas=target
                    ).summary["tail_sensor_mean"]
                    for drive in drives
                ],
            }
        )
        result = calibrate_gas_to_observations(
            config,
            GAS_LIBRARY["argon"],
            observations,
            parameter_names=["ionization_threshold_td"],
            maximum_function_evaluations=20,
        )
        self.assertLess(result.diagnostics["rmse"], 5.0e-3)
        self.assertAlmostEqual(
            result.calibrated_gas.ionization_threshold_td, 125.0, delta=3.0
        )

    def test_reaction_diffusion_output(self) -> None:
        result = simulate_reaction_diffusion_1d(
            replace(self.config, measurement_noise_std=0.0),
            SpatialModelConfig(
                grid_points=31,
                duration_s=0.001,
                dt_s=5.0e-6,
                snapshot_interval_s=2.5e-4,
            ),
        )
        self.assertTrue(
            np.isfinite(result.data["electron_density_m3"].to_numpy()).all()
        )
        self.assertGreater(
            result.summary["center_to_edge_density_ratio"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
