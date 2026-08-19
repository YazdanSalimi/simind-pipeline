# SIMIND Pipeline

Monte Carlo SPECT simulation ([SIMIND](https://simind.blogg.lu.se/)) + quantitative
reconstruction ([PyTomography](https://github.com/PyTomography/PyTomography)) for
generating large, realistic, **multi-isotope, multi-scanner, multi-view-count**
synthetic SPECT datasets from real PET/CT-derived phantoms.

Built for batches spanning hundreds of patients × multiple scanner models ×
view counts × isotopes (Lu-177, Ac-225, Tc-99m) × resample spacings.

- **Every reconstruction algorithm PyTomography ships is supported**: OSEM,
  MLEM, OSMAPOSL, BSREM, RBIEM, RBIMAP, SART, and FBP, plus relative-difference
  and quadratic priors — selected with a single `ALGORITHM`/`PRIOR_TYPE`
  setting, no code changes required.
- **Multiple scanners and multiple isotope emission peaks can be simulated and
  reconstructed together in one run.** `reconstruct_all` takes
  `activity_mbq`/`time_per_proj_s` as a **plain number or a dict keyed by
  isotope** (e.g. `activity_mbq={"lu177": 750, "ac225": 40}`), so a batch
  spanning several tracers gets each isotope's own injected activity and
  acquisition time applied correctly, all reconstructed together in the same
  pass instead of separate runs per isotope.
- **CT input must already be in Hounsfield Units** — `prepare_simind_input`
  converts HU directly to density (g/cm³) via a built-in bilinear HU→ρ curve
  (`density_is_hu=True` by default); don't pre-convert to density yourself.
- **Verified on both Windows and Linux.**
- **Simulation parameters can be changed without rebuilding the dataset.**
  `run_all.sh`/`run_all.bat`/`run_all_mpi.bat` (the MPI-parallel variant) are
  plain text — each line is a standalone `simind`/`simind_mpi` command, so
  things like the `/NN:` history multiplier or the MPI rank count can be
  hand-edited (or swapped for a different run of the same file) without
  regenerating the underlying `.smc`/`.dmi`/`.smi` build files.

## What's actually in here

| Stage | Script | What it does |
|---|---|---|
| **Build** | `prepare-simulation.py` | Resamples PET/CT to each target spacing, then builds every scanner × view × isotope combo's SIMIND input files (`.smc`/`.dmi`/`.smi`/`.win`) + a `run_all.sh` per combo. Two-phase, multiprocessing-parallel (resample once per patient×spacing, reuse across every combo). |
| **Core library** | `simind_yazdan.py` | The engine everything else calls into: `prepare_simind_input`, `reconstruct`, `reconstruct_all`, isotope-aware peak detection, multi-bed stitching, origin/geometry bookkeeping. One self-contained file, pip-installable. |
| **Reconstruct** | `reconstruct-simulations.py` | Local reconstruction pass: scans a tree of finished SIMIND builds, reconstructs every peak PyTomography needs, skips anything already done with the same settings. |

## Install

```bash
pip install git+https://github.com/YazdanSalimi/simind-pipeline.git
```

or from a local clone:

```bash
git clone https://github.com/YazdanSalimi/simind-pipeline.git
cd simind-pipeline
pip install .
```

This installs `simind_yazdan` and its Python dependencies (SimpleITK, PyTorch,
PyTomography, numpy, natsort, tqdm) so `from simind_yazdan import simind_v9`
works anywhere.

**SIMIND itself is not on PyPI** — it's a separate, license-gated Fortran binary
from [Lund University](https://simind.blogg.lu.se/). Install it yourself and make
sure `simind` (and `simind_mpi` if you want MPI-parallel simulation) is on `PATH`
before running anything here.

`prepare-simulation.py` isn't meant to be imported — it's a **copy, edit the
config block at the top, run** tool. Copy it into your working directory and
edit the paths/settings for your own patients/scanners/isotopes.

## Quickstart

```bash
# 1. Build SIMIND inputs for every patient/scanner/view/isotope/spacing combo
#    (edit the CONFIG block at the top of prepare-simulation.py first)
python prepare-simulation.py

# 2. Run the generated run_all.sh for each build to simulate it (uses simind
#    itself; run these however suits your machine — sequentially, in parallel,
#    or via your own job scheduler)

# 3. Reconstruct the finished builds
python reconstruct-simulations.py
```

## Design notes worth knowing

- **Multi-isotope aware.** Lu-177, Ac-225 etc. get one shared SIMIND run per bed
  covering *all* their emission peaks (`combine_windows`), instead of duplicating
  the whole simulation once per peak.
- **Resampling is done once, reused everywhere.** Each patient/spacing
  combination is resampled a single time and reused across every scanner ×
  view × isotope combo built from it, instead of repeating the resample per
  combo.

## Requirements

- Python ≥ 3.9
- SIMIND v9+ (separate install, not on PyPI)
- [PyTomography](https://github.com/PyTomography/PyTomography) (reconstruction engine)
- A CUDA GPU is strongly recommended for reconstruction (PyTomography/PyTorch)

## License

MIT (see `pyproject.toml`) — adjust if you'd rather keep this closed.

## Author

Yazdan Salimi
