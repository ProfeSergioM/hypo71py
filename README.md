<p align="center">
  <img src="assets/image.png" alt="hypo71py logo" width="360">
</p>

# 🐍 hypo71py

**hypo71py** is a Python implementation of the classic **HYPO71** earthquake location algorithm, providing a fast, deterministic, and operationally proven method for local earthquake hypocenter determination.

The package is derived from legacy HYPO71 Fortran code, with the original Python rewriting by **Kris Vanneste** and minor adaptations by **Dan Sandiford**.

The goal of `hypo71py` is to provide a **standalone, modern, and ObsPy-native** implementation of HYPO71 that can be used directly in research workflows, notebooks, and automated catalog pipelines.


---
  

## Installation

### Dependencies

- Python ≥ 3.9  
- numpy  
- scipy  
- obspy  

(Optional: matplotlib, pandas, Fortran compiler for f2py speedups)

### Install (editable)

```bash
git clone https://github.com/<your-org>/hypo71py.git
cd hypo71py
pip install -e .
```

### Fortran speedups (optional but recommended)

The travel-time kernel (`TRVDRV`) and step-regression routine (`SWMREG`) can
be compiled as f2py extensions for a significant speed improvement.
Pre-compiled binaries for **macOS / Python 3.10** are included in the
repository (`hypo71py/core/fortran/*.cpython-310-darwin.so`).  For any other
platform or Python version, recompile from source:

```bash
cd hypo71py/core/fortran
python -m numpy.f2py -c trvdrv.f tinorm.f -m trvdrv
python -m numpy.f2py -c swmreg.f         -m swmreg
```

Requirements: `gfortran` (or another f2py-compatible Fortran compiler) and
`numpy` with f2py support (included in the standard numpy installation).

```bash
# macOS — install gfortran via Homebrew
brew install gcc

# Linux
sudo apt install gfortran   # Debian/Ubuntu
sudo dnf install gcc-gfortran  # Fedora/RHEL
```

After compilation the `.so` files will appear in `hypo71py/core/fortran/` and
will be picked up automatically.  Pass `use_fortran_speedups=True` (the
default) to `SINGLE` to use them, or `False` to fall back to the pure-Python
implementation.

---

## Scope and Intent

`hypo71py` implements the **Geiger-style iterative least-squares inversion**

- travel times and derivatives in a **1-D layered crustal velocity model**
- fast execution suitable for catalog-scale relocation

---

## Design Philosophy

The codebase is structured around a clear separation of concerns:

### 1. `core/` — Numerical Engine (FORTRAN-faithful)

The `core` package contains a **Python translation** of the original HYPO71 Fortran routines, including:

- travel-time computation and derivatives
- azimuthal weighting
- linearized least-squares updates
- optional Fortran speedups via `f2py`

**Design principle:**  
> Numerical equivalence and transparency are prioritised over Pythonic style.

This makes the code:
- auditable against the original Fortran
- suitable as a reference implementation
- a stable foundation for future optimisation (SciPy / Numba / JAX)

---

### 2. `model/` — Domain Objects (Pythonic)

The `model` layer provides structured several lightweight classes used by HYPO71:

- `Station`, `PhasePick`, `StationPhases`
- `CrustalVelocityModel`
- time and type utilities

These classes:
- encapsulate metadata and bookkeeping
- provide convenience methods (e.g. residual tables, azimuthal gap)
- remain independent of ObsPy where possible

**Design principle:**  
> Express seismic concepts clearly, without embedding workflow logic.

---

### 3. `interface/` — Workflow Glue (ObsPy-native)

The `interface` layer connects `hypo71py` to the broader Python seismology ecosystem, in particular **ObsPy**.

Current interfaces support:
- loading stations from StationXML (single file or directory)
- converting ObsPy `Event` objects into HYPO71 inputs
- relocating events and **attaching new ObsPy `Origin` objects**
- relocating entire catalogs while preserving provenance

**Design principle:**  
> Return first-class ObsPy objects, not custom result formats.

This makes `hypo71py` immediately compatible with:
- QuakeML
- SeisBench / PyOcto outputs
- downstream catalog processing and QC tools




---

##  Acknowledgements

This package builds on the original HYPO71 algorithm and legacy Fortran implementations.  
Credit is due to the original authors of HYPO71 and to contributors to earlier Python ports, including *robspy*.


- **Kris Vanneste**

Primary author of core and model elements (https://gitlab-as.oma.be/kris/robspy)

- **Dan Sandiford**

Code reorganisation and Obspy interface:  
