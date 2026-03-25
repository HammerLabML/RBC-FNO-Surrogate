---
license: mit
task_categories:
  - time-series-forecasting
tags:
  - physics
  - simulation
  - fluid-dynamics
  - rayleigh-benard-convection
  - computational-fluid-dynamics
  - neural-operator
  - surrogate-model
pretty_name: Rayleigh-Benard Convection Dataset
size_categories:
  - 1M<n<10M
---

# Rayleigh-Benard Convection (RBC) Dataset

This dataset contains Direct Numerical Simulation (DNS) data of Rayleigh-Benard convection in two and three dimensions, generated using [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). It accompanies the paper [*Fourier neural operators as data-driven surrogates for two- and three-dimensional Rayleigh-Benard convection*](https://doi.org/10.1016/j.neucom.2026.133201), published in Neurocomputing (2026).

## Dataset Description

Rayleigh-Benard Convection describes convection in a fluid layer heated from below and cooled from above. The dataset provides time-series trajectories of the full flow state (velocity, temperature, pressure) at different Rayleigh numbers, suitable for training data-driven surrogate models.

## Dataset Structure

```
2D/
  train/   # 50 episodes per Ra (7 Ra values)
  val/     # 10 episodes per Ra
  test/    # 20 episodes per Ra
3D/
  train/   # 50 episodes at Ra=2500
  val/     # 20 episodes
  test/    # 20 episodes
```

Each file is named `ra{value}.h5` and stored in HDF5 format with the following datasets per episode `i`:

| Key | 2D Shape | 3D Shape | Description |
| --- | --- | --- | --- |
| `states{i}` | `(1000, 5, 64, 96)` | `(1000, 6, 32, 48, 48)` | Flow state: velocity + temperature + pressure |
| `actions{i}` | `(1000, 12)` | `(1000, 8, 8)` | Boundary heating actions |
| `nusselts{i}` | `(1000,)` | `(1000,)` | Nusselt number (convective heat flux) |

### State channels

- **2D** (5 channels): u, v, T, p_hyd, p_non (horizontal velocity, vertical velocity, temperature, hydrostatic pressure, non-hydrostatic pressure)
- **3D** (6 channels): u, v, w, T, p_hyd, p_non (3 velocity components, temperature, hydrostatic pressure, non-hydrostatic pressure)

The last two channels (pressure) are typically excluded during surrogate model training.

## Simulation Parameters

| Parameter | 2D | 3D |
| --- | --- | --- |
| Domain **D** | 2pi x 2 | 4pi x 4pi x 2 |
| Grid **N** | 96 x 64 | 48 x 48 x 32 |
| Ra | {1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7} | 2500 |
| Pr | 0.7 | 0.7 |
| (T_cold, T_hot) | (1, 2) | (0, 1) |
| dt | 0.03 | 0.01 |
| Timesteps per episode | 1000 | 1000 |

## Usage

```python
import h5py

with h5py.File("2D/train/ra300000.h5", "r") as f:
    # Load episode 0 states: (1000, 5, 64, 96)
    states = f["states0"][:]

    # Velocity and temperature (exclude pressure)
    uvT = states[:, :3]  # (1000, 3, 64, 96)
```

For use with the training code, see the [RBC-FNO-Surrogate](https://github.com/tmarkmann/RBC-FNO-Surrogate) repository.

## Citation

```bibtex
@article{MARKMANN2026133201,
  title = {Fourier neural operators as data-driven surrogates
  for two- and three-dimensional Rayleigh-Benard convection},
  journal = {Neurocomputing},
  volume = {679},
  pages = {133201},
  year = {2026},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2026.133201},
  author = {Thorben Markmann and Michiel Straat and Sebastian
  Peitz and Barbara Hammer},
}
```
