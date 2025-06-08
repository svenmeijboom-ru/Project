#  Vicsek Swarm Simulation with Resource-Driven Behavior

This project extends the classical [Vicsek model](https://en.wikipedia.org/wiki/Vicsek_model) by introducing **food-driven urge**. Agents (particles) are attracted to **environmental stimuli** (food), simulating biologically-inspired swarming behavior such as foraging.

---

##  Repository Contents

- `Script.py` – Main Python script implementing the simulation.
- `*.png` – Output graphs (generated after running the experiments).
- `viscek.py` - Initial commit model (obsolete).

---

## Installation

To run the simulation, install the required dependencies:

```bash
pip install numpy scipy matplotlib
```

## Quick Start

```bash
python Script.py
```

The script runs in one of several modes (controlled via `CFG.MODE`):

- `MODE = 1`: Run a single flocking simulation (with optional animation).
- `MODE = 2`: Run experiments over varying food strengths and plot group cohesion.
- `MODE = 3`: Run noise vs. food count experiments and analyze average cohesion.

Edit parameters in the `CFG` class inside `Script.py` to change behaviors like noise, number of food sources, and food strength.

---

##  Configuration Parameters

All tunable settings are located in the `CFG` class in `Script.py`. Key ones include:

| Name                      | Purpose                                             |
|---------------------------|-----------------------------------------------------|
| `N`                       | Number of particles                                 |
| `L`                       | Simulation grid size (L × L)                        |
| `ETA`                     | Noise level in orientation updates                  |
| `R0`                      | Interaction radius                                  |
| `v0`                      | Constant particle speed                             |
| `FOOD`                    | Enable or disable food influence                    |
| `FOOD_STRENGTH`           | Urge agents feel toward food                        |
| `NF`                      | Number of food sources                              |
| `RESPAWN_DELAY`           | Frames before a food source reappears after being eaten |
| `NUM_FRAMES`              | Total simulation length                             |

---

##  Output

- **Animation** (in MODE 1): Shown live using `matplotlib`'s quiver plot.
- **Plots** (in MODE 2 & 3): Saved automatically as `.png` images:
  - `cohesion_graph-NF_*.png`
  - `noise_vs_cohesion-*.png`

---

##  Metrics Tracked

- **Cohesion** – Mean alignment of particle headings over time

---

##  References

- T. Vicsek et al. (1995), *Novel type of phase transition in a system of self-driven particles*
- Francesco Turci’s [Minimal Vicsek Python Model](https://francescoturci.net/2020/06/19/minimal-vicsek-model-in-python/)

---

