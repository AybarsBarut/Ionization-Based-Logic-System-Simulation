# Contributing

Thank you for considering a contribution to the Ionization-Based Logic System
Simulation.

## Development Setup

```bash
python -m venv .venv
python -m pip install -e .
python -m unittest -v
```

## Contribution Guidelines

1. Keep the project educational and software-only.
2. Do not add laboratory operating procedures or high-voltage instructions.
3. Label coefficients as conceptual, empirical, or experimentally calibrated.
4. Explain new equations and parameters in `MODEL_RAPORU.md`.
5. Add focused tests for new physical states, metrics, or logic behavior.
6. Preserve deterministic random seeds in reproducible examples.
7. Avoid presenting simplified outputs as validated laboratory predictions.

## Pull Requests

Before opening a pull request:

```bash
python -m py_compile plasma_model.py advanced_models.py run_experiments.py
python -m unittest -v
python run_experiments.py --output-dir outputs
```

Describe the physical assumption being introduced, the expected qualitative
behavior, and the verification performed.
