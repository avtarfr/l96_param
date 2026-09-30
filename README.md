# L96 Parameterization Project

Online stability of NN parameterizations, data-volume sweep.

## Research Question

What minimal training-data volume is required for a neural network parameterization of the Lorenz-96 subgrid coupling term $U_k$ to remain **stable** and **physically consistent** when coupled online into the reduced (slow-variable only) model — as opposed to merely being accurate offline?

Offline accuracy does not guarantee online stability. This project tests whether that gap shows up as a function of training data volume in the L96 two-tier system, which shall serve as a standard cheap testbed (computationally) for this climate parameterization research.

## Background

Modern climate models are increasingly replacing physics-based subgrid parameterizations with neural networks trained on high-resolution data (need citation). These models often perform well offline but show instability online when coupled to a high-resolution physics informed simulations. This offline/online gap is a known open problem in the climate-ML literature. This project investigates it in the simplest possible setting - the L96 two-tier model.

## Status

#### Done:

* Reproduced the L96 two-tier baseline system (no-parameterization GCM)
* Reproduced offline training pipeline: linear, local FCNN, and non-local FCNN models trained to predict $U_k$ from $X_k$, on a single fixed dataset *(FCNN - Fully Connected Neural Network).*
* Reproduced online coupling pipeline: trained models plugged back into the
  X-only integration, short-horizon comparison against true trajectories *(short-horizon here refers to low number of total time for the model to be ran for)*

#### In-progress / next steps:

* Data-volume sweep: retrain offline models across a range of training-data volumes (currently only one fixed volume has been tested)
* Long-horizon online stability testing: extend online coupling runs to detect drift/blow-up over long simulations, not just short-term error
* Define and compute a quantitative stability metric (climatological mean/variance vs. true, divergence threshold) across data volumes

#### Out of scope / skipped over for now:

* Stochastic parameterizations (deterministic point-prediction only, for now)
* Physics-informed / constrained architectures
* Convolutional or other advanced architectures
* Scaling beyond the L96 toy model

## Repository Structure

```
data/           generated L96 truth trajectories, train/test splits
models/         saved trained model weights (per architecture / data volume)
notebooks/      main working notebooks, numbered in pipeline order
src/            reusable code (L96 integrator, training/coupling utilities)
results/        output plots, stability metrics
papers/         reference papers
```

#### Notebooks:

1. `01_baseline_repro.ipynb` — L96, two-tier system, no parameterization baseline GCM
2. `02_offline_training.ipynb` — offline training of NN parameterizations (linear, local FCNN, non-local FCNN)
3. `03_online_coupling.ipynb` — coupling trained models back into the online simulation, comparison against truth (short-horizon for now)

## Setup

It is recommended that a `conda` environment is set up to duplicate this. To setup (assuming miniconda / conda is set-up and working properly):

1. `git clone github.com/avtarfr/l96_param`
2. Open terminal in the cloned directory and - `conda create -n l96_param python=3.11.15`
3. `conda activate l96_param`
4. `pip install -r requirements.txt`

## References

Key papers this project builds on are listed under `papers/`. Closest methodological
precedents: Wilks (2005), Rasp (2020), Gagne et al. (2020). Testbed code adapted from
the [m2lines L96_demo](https://github.com/m2lines/L96_demo) tutorial series.
