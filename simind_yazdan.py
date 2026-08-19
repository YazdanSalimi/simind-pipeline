r"""
simind_yazdan.py — self-contained SIMIND v9 Lu-177/multi-isotope SPECT pipeline:
build inputs (Monte Carlo prep) + reconstruct (OSEM/etc + multi-bed stitch),
all in ONE file. No other project .py files are required to import/use this.
(Renamed from yazdan.py to avoid colliding with an unrelated yazdan.py elsewhere.)

    from simind_yazdan import simind_v9

    sim = simind_v9(scanner="siemens_symbia_t", smc_dir=r"C:\simind\v9\smc_dir")

    # 1) prepare SIMIND inputs for one patient/scanner/view-count build
    build = sim.prepare_simind_input(
        source_nifti=pet_path, density_nifti=ct_path,
        workdir=r"C:\simind\v9\patients\NEMA\siemens_symbia_t--64--views",
        n_azimuth=64, patient_id="NEMA")
    # ... then actually run the simulation (run_all.sh / run_all_mpi.bat in
    # build.workdir, or the patients-root run_all_patients.bat / run_all_server.sh
    # for many builds at once) ...

    # 2) reconstruct every bed found in that folder for one energy peak
    result = sim.reconstruct(build.workdir, peak="113kev", n_iters=4, n_subsets=8)
    print(result["wholebody"])   # {"NC": path, "AC": path, "SC": path, "ACSC": path}

Formerly split across simind_lu177_v9.py (builder) + reconstruct_lu177_multibed.py
(reconstruction) + this file (thin class wrapper). Consolidated into one file on
2026-08-12 at the user's request; every fix already made in either half (case-
insensitive bed-tag lookup, radius-units auto-detection, per-bed activity-scale
normalization, axial-pad cropping, auto-detected BEDS, combine_windows for
multi-peak isotopes, ...) is preserved unchanged — nothing was reimplemented,
only relocated into this one module.
"""
from __future__ import annotations
import os
import re
import sys
import glob
import json
import math
import hashlib
import getpass
import platform
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch

from pytomography.io.SPECT import simind
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.projectors import ExtendedSystemMatrix
from pytomography.transforms.SPECT import (
    SPECTAttenuationTransform, SPECTPSFTransform)
from pytomography.algorithms import (
    OSEM, MLEM, OSMAPOSL, BSREM, RBIEM, RBIMAP, SART, FilteredBackProjection)
from pytomography.priors import RelativeDifferencePrior, QuadraticPrior
from pytomography.likelihoods import PoissonLogLikelihood

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ============================================================================
#  SECTION 1 — BUILDER  (was simind_lu177_v9.py)
# ============================================================================
"""
SIMIND v9 Lu-177 SPECT input builder — writes valid SMCV2 .smc + .dmi/.smi
voxel maps + .win energy windows, from co-registered NIfTI source & density.

Verified against the SIMIND v9 distribution (smc_win_intel_64_v90):
  * .smc is the SMCV2 text format; this module's reader/writer round-trips
    every v9 example file byte-for-byte.
  * Index meanings taken from smc_dir/change.txt (the official v9 index doc),
    NOT from memory. Key voxel-phantom facts encoded here:

  Index 14  = phantom type; -1 => Integer*2 density map (*.dmi), values = 1000*density
  Index 15  = source type;  -1 => Integer*2 source  map (*.smi), 2-byte ints
  Index 31  = voxel side length of the maps [cm]  (NOT the image pixel size)
  Index 78  = density-map matrix size, I (columns)
  Index 79  = source-map  matrix size, I (columns)
  Index 81  = density-map matrix size, J (rows); 0 => copies index 78
  Index 82  = source-map  matrix size, J (rows); 0 => copies index 79
  Index 32  = map slice orientation (0 = transaxial, (i,j)->(y,z))
  Index 33  = first image index in the map file
  Index 34  = number of density maps to read (= n slices)
  Index 35  = density threshold defining the phantom border [g/cm^3]
  Index  5  = HALF axial length of the whole voxel stack [cm]
              (per doc: for voxel phantoms index 5 with n slices sets slice
               thickness = 2*index5 / nslices)
  Index  2  = same, for the source stack half axial length [cm]
  Index  1  = photopeak energy [keV]  (one per run; we emit 113 and 208)
  Index  9  = crystal thickness [cm]
  Index 10  = detector width [cm] (full); index 8 = crystal x length
  Index 12  = radius of rotation: origin -> lowest detector part [cm]
  Index 20/21 = upper/lower energy window (negative => percent window)
  Index 22  = energy resolution %FWHM @140keV ; Index 23 = intrinsic spatial FWHM cm
  Index 26  = histories/1000 per projection — IGNORED for voxel sources;
              instead sum of source-map voxels sets it, scaled by /NN:n
  Index 28  = image pixel size [cm]
  Index 29  = number of projection angles (azimuth)
  Index 30  = rotation mode/direction (0 => 360 CCW)
  Index 76/77 = projection image matrix (cols/rows)
  Index 84  = scoring routine; 1 => Scattwin (needs the .win file)

Both Lu-177 photopeaks (113 & 208 keV) are produced as separate runs, each
with its own .smc (index 1) while the .win file scores both windows.

FOV handling: if the patient axial extent exceeds the detector FOV, the stack
is split into overlapping axial segments; each segment gets its own maps and
its own index-5/index-34 values so distances stay physically correct.

PROVENANCE: every call to build_simind_lu177 writes a JSON log
(build_log_<timestamp>.json) into the workdir capturing all arguments, the
resolved scanner/collimator, the exact index/flag values written into each
.smc, per-segment geometry, and file checksums, so any produced .smc can be
traced back to how it was generated.
"""


# ------------------------------------------------------------------ SMCV2 I/O
class SMC:
    """Read/edit/write a SMCV2 .smc file, preserving exact byte layout."""
    def __init__(self, path: str):
        raw = open(path, "rb").read().decode("latin-1")
        lines = raw.split("\n")
        assert lines[0] == "SMCV2", f"not SMCV2: {lines[0]!r}"
        self.magic, self.title = lines[0], lines[1]
        f = lambda t: next(i for i, l in enumerate(lines) if t in l)
        ib, ifl = f("# Basic Change"), f("# Simulation flags")
        it, idf = f("# Text Variables"), f("# Data files")
        self._h_basic, self._h_flags = lines[ib], lines[ifl]
        self._h_text, self._h_files = lines[it], lines[idf]
        self.n_basic = int(lines[ib].split()[0])
        self.vals = []
        for l in lines[ib + 1:ifl]:
            self.vals += [float(x) for x in re.findall(r"[-+]?\d\.\d+E[-+]\d+", l)]
        self.flags = lines[ifl + 1]
        self.n_text = int(lines[it].split()[0])
        self.text = [lines[it + 1 + k] for k in range(self.n_text)]
        self.n_files = int(lines[idf].split()[0])
        self.files = [lines[idf + 1 + k] for k in range(self.n_files)]
        self._trailing = lines[idf + 1 + self.n_files:]
        # provenance: remember every set()/set_flag() call on this instance
        self._log_indices: dict[int, float] = {}
        self._log_flags: dict[int, bool] = {}

    def set(self, i1, v):
        self.vals[i1 - 1] = float(v)
        self._log_indices[i1] = float(v)
    def get(self, i1):    return self.vals[i1 - 1]
    def set_flag(self, i1, on: bool):
        fl = list(self.flags)
        while len(fl) < i1: fl.append("F")
        fl[i1 - 1] = "T" if on else "F"
        self.flags = "".join(fl)
        self._log_flags[i1] = bool(on)
    def set_collimator(self, name): self.text[0] = f"{name:<10}"[:10]
    def set_file(self, slot, name): self.files[slot] = f"{name:<60}"[:60]

    @staticmethod
    def _fmt(x):
        if x == 0.0: return " 0.00000E+00"
        s = f"{abs(x): .5E}"; m, e = s.split("E")
        d = m.strip().replace(".", "")
        return f"{'-' if x < 0 else ' '}0.{d[:5]}E{int(e)+1:+03d}"

    def serialise(self):
        L = [self.magic, self.title, self._h_basic]
        for i in range(0, self.n_basic, 5):
            L.append("".join(self._fmt(self.vals[j])
                              for j in range(i, min(i + 5, self.n_basic))))
        L += [self._h_flags, self.flags, self._h_text, *self.text,
              self._h_files, *self.files, *self._trailing]
        return "\n".join(L).encode("latin-1")

    def write(self, path): open(path, "wb").write(self.serialise())

    def provenance(self):
        """The exact index values and flags this instance set, for the log."""
        return {
            "title": self.title.strip(),
            "collimator": self.text[0].strip() if self.text else None,
            "indices_set": {str(k): self._log_indices[k]
                            for k in sorted(self._log_indices)},
            "flags_set": {str(k): self._log_flags[k]
                          for k in sorted(self._log_flags)},
            "flags_string": self.flags,
        }


# ------------------------------------------------------------- scanner table
@dataclass
class ScannerSpec:
    name: str
    crystal_thickness_cm: float      # index 9
    detector_x_cm: float             # index 8  (crystal x length)
    detector_width_cm: float         # index 10 (full width)
    energy_res_pct: float            # index 22  %FWHM @140keV
    intrinsic_fwhm_cm: float         # index 23
    default_collimator: str
    collimators: dict = field(default_factory=dict)
    note: str = ""

# collimator names must exist in your SIMIND collimator list. The v9 tutorial
# ships gv-* / si-* / gi-* names. Values marked VERIFY need a data-sheet check.
# Scanner / collimator database.
#
# IMPORTANT: the collimator CODE (e.g. "si-melp") must exist in your install's
# collim.col (in SMC_DIR). Codes below follow SIMIND's standard naming but MUST
# be verified against YOUR collim.col — use list_collimators() (below) to dump
# the real codes on your machine. Detector dims (crystal/detector size) are
# typical published values and marked VERIFY; they affect solid-angle/geometry
# but the collimator code is what matters most for resolution/sensitivity.
#
# Collimator code convention in collim.col: <vendor><-><type>, e.g.
#   si-*  Siemens Symbia/Intevo    ge-* / gv-*  GE Infinia/Discovery/NM
#   ph-*  Philips (BrightView etc) me-*  MiE      to-*  Toshiba/Canon
# Types: lehr (low-E high-res), legp/lego (low-E general), melp/megp (medium-E),
#        he/hegp/heap (high-E), me (medium-E).
SCANNER_DB = {
    # ---- Siemens ------------------------------------------------------- #
    "siemens_intevo": ScannerSpec(
        "Siemens Symbia Intevo", 0.9525, 59.1/2, 44.5, 9.0, 0.38,   # VERIFY
        "melp", {"lehr": "si-lehr", "lego": "si-lego", "melp": "si-melp",
                 "he": "si-he"},
        "Lu-177: use MELP (208 keV penetrates LEHR)."),
    "siemens_symbia_t": ScannerSpec(
        "Siemens Symbia T/T2/T6", 0.9525, 59.1/2, 44.5, 9.0, 0.38,  # VERIFY
        "melp", {"lehr": "si-lehr", "lego": "si-lego", "melp": "si-melp",
                 "he": "si-he", "heap": "si-heap"},
        "I-131/Lu-177 high-E: use HE/HEAP."),
    "siemens_ecam": ScannerSpec(
        "Siemens e.cam", 0.9525, 53.3/2, 38.7, 9.5, 0.40,           # VERIFY
        "lehr", {"lehr": "si-lehr", "melp": "si-melp", "he": "si-he"},
        ""),
    # ---- GE ------------------------------------------------------------ #
    "ge_nm870": ScannerSpec(
        "GE NM/CT 870 DR", 0.9525, 54.0/2, 40.0, 9.0, 0.37,         # VERIFY
        "megp", {"lehr": "ge-lehr", "legp": "ge-legp", "megp": "ge-megp",
                 "hegp": "ge-hegp"},
        "Lu-177: use MEGP."),
    "ge_discovery_670": ScannerSpec(
        "GE Discovery NM/CT 670", 0.9525, 54.0/2, 40.0, 9.0, 0.37,  # VERIFY
        "megp", {"lehr": "ge-lehr", "legp": "ge-legp", "megp": "ge-megp",
                 "hegp": "ge-hegp"},
        "Lu-177: use MEGP."),
    "ge_infinia": ScannerSpec(
        "GE Infinia (Hawkeye)", 0.9525, 54.0/2, 40.0, 9.8, 0.38,    # VERIFY
        "megp", {"lehr": "ge-lehr", "legp": "ge-legp", "megp": "ge-megp",
                 "hegp": "ge-hegp"},
        ""),
    "ge_millennium": ScannerSpec(
        "GE Millennium MG/VG", 0.9525, 54.0/2, 40.0, 10.0, 0.40,    # VERIFY
        "megp", {"lehr": "ge-lehr", "legp": "ge-legp", "megp": "ge-megp",
                 "hegp": "ge-hegp"},
        ""),
    "ge_starguide": ScannerSpec(
        "GE StarGuide (CZT)", 0.5, 51.0/2, 40.0, 6.0, 0.20,         # VERIFY CZT
        "megp", {"lehr": "ge-lehr", "megp": "ge-megp", "hegp": "ge-hegp"},
        "CZT digital detector; energy resolution much better than NaI."),
    # ---- Philips ------------------------------------------------------- #
    "philips_brightview": ScannerSpec(
        "Philips BrightView (X/XCT)", 0.9525, 54.0/2, 40.6, 9.4, 0.35,  # VERIFY
        "megp", {"lehr": "ph-lehr", "legp": "ph-legp", "megp": "ph-megp",
                 "hegp": "ph-hegp"},
        "Lu-177: use MEGP."),
    "philips_forte": ScannerSpec(
        "Philips Forte/Skylight", 0.9525, 54.0/2, 40.6, 9.5, 0.38,  # VERIFY
        "megp", {"lehr": "ph-lehr", "megp": "ph-megp", "hegp": "ph-hegp"},
        ""),
    # ---- Mediso / MiE / Others ---------------------------------------- #
    "mediso_anyscan": ScannerSpec(
        "Mediso AnyScan", 0.9525, 54.0/2, 40.0, 9.0, 0.37,          # VERIFY
        "megp", {"lehr": "me-lehr", "megp": "me-megp", "hegp": "me-hegp"},
        ""),
    "generic_nai": ScannerSpec(
        "Generic NaI SPECT", 0.9525, 54.0/2, 40.0, 9.5, 0.38,       # generic
        "megp", {"lehr": "ge-lehr", "megp": "ge-megp", "hegp": "ge-hegp"},
        "Generic fallback; override collimator= with a code from your "
        "collim.col via list_collimators()."),
}


def list_collimators(smc_dir=None):
    """Dump every collimator code (and its parameters) from YOUR collim.col.

    This is the AUTHORITATIVE list for your install — the codes in SCANNER_DB
    must exist here. Pass smc_dir or set the SMC_DIR env var; falls back to the
    usual Windows path. Returns a list of (code, params_line) and prints them.
    """
    dirs = [d for d in [smc_dir, os.environ.get("SMC_DIR"),
                        r"C:\simind\v9\smc_dir"] if d]
    path = None
    for d in dirs:
        cand = os.path.join(d, "collim.col")
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        print(f"collim.col not found in {dirs}")
        return []
    out = []
    for line in open(path, "r", errors="ignore"):
        s = line.strip()
        # collimator entries start with '*' then the code (per the manual)
        if s.startswith("*"):
            parts = s[1:].split(None, 1)
            code = parts[0].lower()
            out.append((code, s))
    print(f"{len(out)} collimators in {path}:")
    for code, ln in out:
        print(f"  {code:12s} {ln}")
    return out

LU177_PEAKS = [(113.0, 20.0), (208.0, 20.0)]   # (center keV, width %)

# ---------------------------------------------------------- isotope registry
# Per radionuclide: standard imaging photopeaks (center keV, window width %),
# the .isd spectrum base name (must exist in SMC_DIR), and the recommended
# collimator TYPE key (looked up in the scanner's collimator dict).
#
# Window widths are typical clinical values; adjust per your protocol. The
# collimator recommendation reflects the dominant photon energy:
#   Tc-99m 140 keV      -> LEHR (low energy)
#   Lu-177 113/208 keV  -> MEGP/MELP (medium energy; 208 penetrates LEHR)
#   I-131  364 keV      -> HEGP/HE (high energy)
#   Ac-225 (daughters)  -> imaged via daughter photons; 218 keV (Fr-221) and
#                          440 keV (Bi-213) are the practical windows -> ME/HE.
ISOTOPES = {
    "tc99m": {
        "isd": "tc99m",
        "peaks": [(140.5, 15.0)],
        "collimator": "lehr",
        "note": "Tc-99m 140 keV, LEHR.",
    },
    "lu177": {
        "isd": "lu177",
        "peaks": [(113.0, 20.0), (208.0, 20.0)],
        "collimator": "megp",           # MELP/MEGP
        "note": "Lu-177 113 & 208 keV, medium-energy collimator.",
    },
    "i131": {
        "isd": "i131",
        "peaks": [(364.5, 20.0)],
        "collimator": "hegp",           # high energy
        "note": "I-131 364 keV, high-energy collimator.",
    },
    "ac225": {
        "isd": "ac225",
        # Ac-225 itself emits few useful gammas; imaging uses daughter photons.
        # Fr-221 ~218 keV and Bi-213 ~440 keV are the practical windows.
        "peaks": [(218.0, 20.0), (440.0, 20.0)],
        "collimator": "hegp",           # 440 keV needs high energy
        "note": "Ac-225 imaged via daughters: Fr-221 218 keV, Bi-213 440 keV. "
                "440 keV requires HE collimator; verify ac225.isd exists.",
    },
}


def isotope_config(name):
    """Return the registry entry for an isotope name (case-insensitive)."""
    key = name.lower().replace("-", "").replace("_", "")
    if key not in ISOTOPES:
        raise KeyError(f"isotope '{name}' not in registry {list(ISOTOPES)}. "
                       f"Add it to ISOTOPES with its peaks and .isd name.")
    return ISOTOPES[key]


@dataclass
class BuildResult:
    workdir: str
    segments: list
    commands: list
    win_file: str
    log_file: str = ""          # path to the JSON provenance log
    # {peak_keV: {"peak": {"lo_keV":, "hi_keV":}, "lower": {...}, "upper": {...}}}
    # "lower"/"upper" (TEW scatter windows) are only present if tew_scatter
    # was True. Same numbers as in log_file's resolved.win_rows_per_peak,
    # just reshaped by role for direct lookup without re-reading the log.
    windows: dict = field(default_factory=dict)


# ------------------------------------------------------------------ helpers
def _read(path):
    im = sitk.ReadImage(path)
    return im, sitk.GetArrayFromImage(im), im.GetSpacing()  # arr = (z,y,x)


def _sha256(path, limit_mb=None):
    """SHA-256 of a file. limit_mb caps how much is hashed (None = whole)."""
    h = hashlib.sha256()
    cap = None if limit_mb is None else int(limit_mb * 1024 * 1024)
    read = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            if cap is not None and read + len(chunk) > cap:
                h.update(chunk[:cap - read]); break
            h.update(chunk); read += len(chunk)
    return h.hexdigest()


def _map_to_nifti(arr_zyx, out_path, origin_mm_xyz, spacing_mm_xyz, direction9):
    """Write a SIMIND map array (z,y,x, the exact int16 values SIMIND reads)
    to a spatially-registered NIfTI so it overlays on the original PET/CT.
    arr_zyx is the numpy array in (z,y,x) order (as sliced from the source)."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr_zyx))
    img.SetOrigin([float(o) for o in origin_mm_xyz])
    img.SetSpacing([float(s) for s in spacing_mm_xyz])
    img.SetDirection([float(d) for d in direction9])
    sitk.WriteImage(img, out_path)
    return out_path


INTERP_METHODS = {
    "linear": sitk.sitkLinear,
    "nearest": sitk.sitkNearestNeighbor,
    "bspline": sitk.sitkBSpline,
}


def _resolve_interp(name):
    key = str(name).strip().lower()
    if key not in INTERP_METHODS:
        raise ValueError(
            f"interp must be one of {list(INTERP_METHODS)}, got {name!r}")
    return INTERP_METHODS[key]


def resample_to_spect_grid(img, new_spacing_mm=4.0, interp="linear",
                           default_value=0.0):
    """Resample a SimpleITK image to an isotropic SPECT-scale grid.

    512x512 CT/PET grids are far too fine for SPECT Monte Carlo (huge memory,
    slow, and physical SPECT resolution is ~1 cm anyway). 4 mm is a sane default.

    interp: "linear" (default) blends voxel values across edges -- e.g. a
        phantom's perfectly uniform hot sphere picks up a ramp of
        intermediate values at its boundary instead of a sharp step.
        "nearest" keeps exact input values with a stepped/aliased boundary
        instead of a blended one -- use this when you need the ACTUAL
        simulated value at a location, not an interpolation artifact (e.g.
        validating a NEMA/Jaszczak sphere's true activity concentration).
        "bspline" is a smoother higher-order interpolation than linear
        (less aliasing on fine structure) at extra compute cost.
    """
    old_sp = np.array(img.GetSpacing()); old_sz = np.array(img.GetSize())
    new_sp = np.array([float(new_spacing_mm)] * 3)
    new_sz = np.maximum(1, np.round(old_sz * old_sp / new_sp)).astype(int).tolist()
    method = _resolve_interp(interp)
    return sitk.Resample(img, new_sz, sitk.Transform(), method,
                         img.GetOrigin(), new_sp.tolist(), img.GetDirection(),
                         default_value, img.GetPixelID())


def hu_to_density(hu_array):
    """Bilinear HU -> density (g/cm^3). Generic; swap in your scanner's own
    calibration curve if you have one."""
    hu = hu_array.astype(np.float32)
    rho = np.where(hu < 0.0, 1.0 + hu / 1000.0, 1.0 + hu / 1900.0)
    return np.clip(rho, 0.0, 3.0)


def prepare_inputs(source_nifti, density_nifti, workdir, *,
                   density_is_hu=True, resample_spacing_mm=4.0,
                   pet_interp="linear", ct_interp="linear"):
    """Resample source+density to a common SPECT grid and (optionally) convert
    an HU density map to g/cm^3. Density is resampled ONTO the source grid so
    the two are guaranteed identical in size/spacing/origin. Returns (src, den)
    paths to the prepared NIfTIs.

    pet_interp / ct_interp: "linear" (default), "nearest", or "bspline" --
    independently choose how the activity (PET/source) map and the density
    (CT) map are each resampled onto the SPECT grid. Linear/bspline blend
    voxel values across edges -- e.g. a phantom's perfectly uniform hot
    sphere picks up a ramp of intermediate values at its boundary instead of
    a sharp step. Use "nearest" on either one to keep exact input values
    (stepped/aliased boundary instead of blended) -- useful for quantitative
    phantom validation where you want the actual simulated value, not an
    interpolation artifact. See resample_to_spect_grid()'s docstring for
    more detail on the trade-offs between the three.
    """
    os.makedirs(workdir, exist_ok=True)
    src = sitk.ReadImage(source_nifti)
    den = sitk.ReadImage(density_nifti)

    src_r = resample_to_spect_grid(src, resample_spacing_mm,
                                   interp=pet_interp, default_value=0.0)
    den_default = -1000.0 if density_is_hu else 0.0
    den_r = sitk.Resample(den, src_r, sitk.Transform(), _resolve_interp(ct_interp),
                          den_default, den.GetPixelID())

    if density_is_hu:
        rho = hu_to_density(sitk.GetArrayFromImage(den_r))
        den_out = sitk.GetImageFromArray(rho)
        den_out.CopyInformation(src_r)
    else:
        den_out = den_r

    sp = os.path.join(workdir, "source_prepared.nii.gz")
    dp = os.path.join(workdir, "density_prepared.nii.gz")
    sitk.WriteImage(src_r, sp)
    sitk.WriteImage(den_out, dp)
    return sp, dp

def _segments(nz, sz_cm, fov_cm, overlap_cm):
    if nz * sz_cm <= fov_cm + 1e-6:
        return [(0, nz)]
    seg = max(1, int(round(fov_cm / sz_cm)))
    step = max(1, int(round((fov_cm - overlap_cm) / sz_cm)))
    out, s = [], 0
    while s < nz:
        e = min(s + seg, nz); out.append((s, e))
        if e >= nz: break
        s += step
    return out


def build_simind_lu177(
    source_nifti: str, density_nifti: str, template_smc: str,
    scanner: str, workdir: str, *,
    collimator: Optional[str] = None,
    n_azimuth: int = 120,
    matrix_size: Optional[int] = None,       # image matrix; None => auto-fit map
    auto_matrix: bool = True,                # size image matrix >= map matrix
    pixel_size_cm: Optional[float] = None,   # image pixel; default = voxel size
    photon_multiplier_nn: int = 1,           # /NN:n history multiplier
    source_hist_max: float = 1000.0,         # histories emitted by brightest
                                             # voxel; lower => far faster sim.
                                             # 1000 = high quality/dosimetry;
                                             # ~50-100 = fast phantom tests.
    energy_windows: Optional[list] = None,
    radius_of_rotation_cm: float = 25.0,     # index 12
    fov_axial_cm: float = 40.0,
    fov_overlap_cm: float = 5.0,
    rotation_mode: int = 0,                  # index 30: 0 => 360 CCW
    density_border_threshold: float = 0.05,  # index 35 [g/cm^3]
    isotope: Optional[str] = None,           # e.g. "lu177" -> use full spectrum
                                             # (.isd) via negative Index 1 + /fi.
                                             # None => monoenergetic at each peak.
    tew_scatter: bool = True,                # add lower/upper TEW scatter windows
    tew_width_pct: float = 4.0,              # scatter window half-width (% of peak)
    smc_dir: Optional[str] = None,           # SMC_DIR path (for locating .isd);
                                             # falls back to env + hardcoded path
    log_file: Optional[str] = None,          # override JSON log path;
                                             # defaults to workdir\simulation-debug.json
    save_debug_nifti: bool = True,           # also write .dmi/.smi as NIfTI
    on_line_printout: bool = False,          # SMC Flag-1: verbose on-line
                                             # printout + results to the
                                             # terminal during the run. The
                                             # SIMIND manual explicitly says
                                             # to turn this OFF for batch-
                                             # queue submissions -- default
                                             # here reflects that. Set True
                                             # for an interactive/small test
                                             # build where you want to watch
                                             # live output.
    mpi_ranks: int = 14,                     # -n for run_all_mpi.bat
    simind_bin_dir: Optional[str] = None,    # if set, run_all.sh gets an
                                             # `export PATH="<this>:$PATH"`
                                             # line prepended (before the
                                             # `cd`) so it finds `simind` on
                                             # its own -- e.g. for a Linux/
                                             # HPC install where `simind`
                                             # isn't already on PATH.
                                             # None (default) -> run_all.sh
                                             # unchanged, caller must have
                                             # simind on PATH already.
    patient_id: Optional[str] = None,        # segment tags become
                                             # "{patient_id}_00", "{patient_id}_01",
                                             # ... instead of "seg00", "seg01".
                                             # Defaults to workdir's parent folder
                                             # name (patients/<patient_id>/<...>).
) -> BuildResult:
    """Build the v9 SIMIND input set for a Lu-177 voxel-phantom SPECT sim.

    source_nifti  : activity map (arbitrary units; relative counts).
    density_nifti : density map in g/cm^3 (e.g. from CT via HU->rho).
    template_smc  : any valid v9 .smc (e.g. the shipped simind.smc); used as base.
    Returns BuildResult with one command per (segment x photopeak).

    Writes a JSON provenance log to workdir\simulation-debug.json (path also
    returned in BuildResult.log_file) recording every argument and every
    index/flag written, plus an owner/date record.
    """
    if scanner not in SCANNER_DB:
        raise KeyError(f"scanner must be one of {list(SCANNER_DB)}")
    spec = SCANNER_DB[scanner]

    # Resolve isotope config: peaks, collimator, and .isd spectrum name.
    # Explicit arguments (energy_windows, collimator) override the registry.
    iso_cfg = isotope_config(isotope) if isotope else None
    if iso_cfg:
        peaks = energy_windows or iso_cfg["peaks"]
        col_type = collimator or iso_cfg["collimator"]
        isd_name = iso_cfg["isd"]
    else:
        peaks = energy_windows or LU177_PEAKS
        col_type = collimator or spec.default_collimator
        isd_name = isotope   # may be None (monoenergetic)

    # map the collimator TYPE to this scanner's actual code. Vendors name the
    # same energy class differently (Siemens ME=melp, GE ME=megp; high-E he vs
    # hegp vs heap), so try equivalents before failing.
    EQUIV = {
        "megp": ["megp", "melp", "me", "megp"],
        "melp": ["melp", "megp", "me"],
        "me":   ["me", "melp", "megp"],
        "hegp": ["hegp", "he", "heap", "hego"],
        "he":   ["he", "hegp", "heap"],
        "lehr": ["lehr", "lego", "legp"],
        "legp": ["legp", "lego", "lehr"],
        "lego": ["lego", "legp", "lehr"],
    }
    candidates = EQUIV.get(col_type, [col_type])
    col_type_resolved = next((c for c in candidates if c in spec.collimators),
                             None)
    if col_type_resolved is None:
        raise KeyError(
            f"collimator '{col_type}' (or equivalents {candidates}) not "
            f"available for {scanner}. Options: {list(spec.collimators)}. "
            f"Add it to SCANNER_DB or pass collimator= with a valid type, and "
            f"verify the code exists via list_collimators().")
    col_name = spec.collimators[col_type_resolved]
    os.makedirs(workdir, exist_ok=True)

    # Segment tags default to "{patient_id}_00", "{patient_id}_01", ... so
    # multi-patient outputs (and downstream file names) are identifiable
    # without cross-referencing the workdir path. If not given explicitly,
    # derive it from the workdir layout this repo uses everywhere:
    # patients/<patient_id>/<scanner>--<views>--views, i.e. the parent folder.
    pid = patient_id or os.path.basename(os.path.dirname(os.path.abspath(workdir)))
    pid = pid or "seg"   # last-resort fallback if workdir has no parent segment

    # ---- start the provenance record (arguments + environment) ------------ #
    started = datetime.datetime.now().astimezone()
    log = {
        "schema": "simind_lu177_build_log/v1",
        "created": started.isoformat(),
        "owner": {"name": "Yazdan Salimi", "email": "salimiyazdan@gmail.com"},
        "environment": {
            "user": getpass.getuser(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "simpleitk": sitk.__version__,
            "builder_file": os.path.abspath(__file__),
            # NOT auto-queried (would mean spawning simind.exe just for this).
            # This install (C:\simind\v9\simind.exe) reports "V9.0" in its
            # banner and in every Interfile header it writes. Recorded here
            # because SIMIND's Interfile "Radius" field's units depend on
            # this: confirmed v9.0 writes Radius in mm, while older v8.0
            # output (e.g. reference/tutorial data from other installs)
            # writes it already in cm. get_metadata() defaults to assuming
            # cm, so reading v9.0 output without accounting for this gives a
            # 10x-wrong object-to-collimator distance -> a massively
            # oversized PSF blur that smears the whole reconstruction into
            # one indistinct blob (seen firsthand on the NEMA phantom).
            # radius_distance_unit() (below) already auto-detects this from
            # the header value itself rather than trusting a hardcoded
            # version, so no action is needed if this install is ever
            # upgraded -- this note is for humans, not code.
            "simind_exe_version_seen": "V9.0",
        },
        "arguments": {
            "source_nifti": os.path.abspath(source_nifti),
            "density_nifti": os.path.abspath(density_nifti),
            "template_smc": os.path.abspath(template_smc),
            "scanner": scanner,
            "workdir": os.path.abspath(workdir),
            "collimator": collimator,
            "n_azimuth": n_azimuth,
            "matrix_size": matrix_size,
            "pixel_size_cm": pixel_size_cm,
            "photon_multiplier_nn": photon_multiplier_nn,
            "source_hist_max": source_hist_max,
            "energy_windows": energy_windows,
            "radius_of_rotation_cm": radius_of_rotation_cm,
            "fov_axial_cm": fov_axial_cm,
            "fov_overlap_cm": fov_overlap_cm,
            "rotation_mode": rotation_mode,
            "density_border_threshold": density_border_threshold,
            "isotope": isotope,
            "simind_bin_dir": simind_bin_dir,
            "on_line_printout": on_line_printout,
        },
        "resolved": {
            "scanner_spec": asdict(spec),
            "collimator_code": col_name,
            "peaks_center_width": [list(p) for p in peaks],
        },
        "inputs": {},          # source/density file digests + geometry
        "segments": [],        # per-segment maps + per-smc index/flag record
    }

    src_img, src, sp = _read(source_nifti)
    _, den, sp_d = _read(density_nifti)
    if src.shape != den.shape:
        raise ValueError(f"grid mismatch {src.shape} vs {den.shape}")
    # spacing (x,y,z) mm -> cm ; require ~isotropic in-plane for voxel side
    sx, sy, sz = (s / 10.0 for s in sp)
    nz, ny, nx = src.shape
    voxel_cm = sx
    if not (math.isclose(sx, sy, rel_tol=1e-3)):
        # SIMIND uses a single voxel side (index 31); warn via note in title.
        voxel_cm = (sx + sy) / 2.0
    pix = pixel_size_cm or voxel_cm

    # ---- image matrix sizing -------------------------------------------- #
    # The projection/image matrix (index 76/77) must be large enough that the
    # in-plane object (nx,ny at voxel_cm) is not truncated. Physical object
    # width = max(nx,ny)*voxel_cm; it must fit inside matrix*pix.
    # If matrix*pix < object width, the reconstruction is cropped/distorted
    # (this was the 137-map-into-128-image truncation).
    map_in_plane = max(nx, ny)
    if matrix_size is None:
        if auto_matrix:
            # smallest even matrix >= map so the object always fits at this pix
            need = math.ceil(map_in_plane * voxel_cm / pix)
            matrix_size = int(need + (need % 2))          # keep it even
        else:
            matrix_size = 128
    else:
        # user forced a matrix; check it doesn't truncate and warn/raise
        covered_cm = matrix_size * pix
        object_cm = map_in_plane * voxel_cm
        if covered_cm < object_cm - 1e-6:
            if auto_matrix:
                # honor the intent to avoid truncation: grow to fit
                need = math.ceil(object_cm / pix)
                matrix_size = int(need + (need % 2))
            else:
                raise ValueError(
                    f"matrix_size={matrix_size} at pixel {pix} cm covers "
                    f"{covered_cm:.1f} cm, but the object is {object_cm:.1f} cm "
                    f"wide ({map_in_plane} vox * {voxel_cm} cm). It would be "
                    f"truncated. Increase matrix_size to >= "
                    f"{math.ceil(object_cm/pix)}, raise pixel_size_cm, or set "
                    f"auto_matrix=True.")

    # ---- axial (J / index-77) matrix sizing ----------------------------- #
    # The AXIAL extent of each bed is the SPECT scan range (fov_axial_cm), NOT
    # the in-plane matrix. SIMIND supports a rectangular projection image:
    # index 76 = image cols (I, in-plane) and index 77 = image rows (J, axial)
    # are independent (verified in smc_dir/change.txt). So the detector has
    # `axial_matrix` rows and PyTomography reconstructs an object that many
    # voxels tall. axial_matrix = fov_axial_cm / pix (e.g. 40 cm / 0.5 cm = 80),
    # kept even. This removes the old need to pad each bed up to matrix_size:
    # the object and its attenuation map are now both `axial_matrix` tall by
    # construction, so no air padding is added and stitched beds carry only the
    # real 40 cm of data.
    axial_matrix = int(round(fov_axial_cm / pix))
    axial_matrix += axial_matrix % 2                  # keep it even
    log["resolved"] = log.get("resolved", {})
    log["resolved"]["axial_matrix"] = int(axial_matrix)
    log["resolved"]["patient_id"] = pid

    # Full spatial frame of the PREPARED source (the grid the maps were carved
    # from). This is what lets a reconstructed bed be placed back into patient
    # world coordinates. SimpleITK origin/spacing are in mm, (x,y,z) order.
    src_origin_mm = list(src_img.GetOrigin())        # (x,y,z) mm
    src_spacing_mm = list(src_img.GetSpacing())      # (x,y,z) mm
    src_direction = list(src_img.GetDirection())     # 9-element row-major

    log["inputs"] = {
        "source_nifti_sha256": _sha256(source_nifti),
        "density_nifti_sha256": _sha256(density_nifti),
        "template_smc_sha256": _sha256(template_smc),
        "array_shape_zyx": [int(nz), int(ny), int(nx)],
        "spacing_cm_xyz": [sx, sy, sz],
        "voxel_side_cm_used": voxel_cm,
        "image_pixel_size_cm": pix,
        "in_plane_isotropic": bool(math.isclose(sx, sy, rel_tol=1e-3)),
        "map_in_plane": int(map_in_plane),
        "image_matrix_size": int(matrix_size),
        "image_fov_cm": matrix_size * pix,
        "object_width_cm": map_in_plane * voxel_cm,
        # world-frame of the prepared source (SimpleITK conventions, mm, xyz)
        "prepared_source_origin_mm_xyz": src_origin_mm,
        "prepared_source_spacing_mm_xyz": src_spacing_mm,
        "prepared_source_direction": src_direction,
    }

    segs = _segments(nz, sz, fov_axial_cm, fov_overlap_cm)
    log["resolved"]["segment_ranges_z"] = [[int(a), int(b)] for a, b in segs]

    # ---- .win files (scattwin format) ------------------------------------ #
    # combine_windows: for a multi-peak ISOTOPE build (isotope= set, e.g.
    # Lu-177's 113+208 keV, Ac-225's 218+440 keV daughters), SIMIND already
    # simulates the isotope's FULL emission spectrum regardless of which
    # window you score -- confirmed empirically (2026-08-12, NEMAIEC lu177
    # 128-view build): running the 113 keV .smc unchanged but pointing it at
    # a combined 6-row .win file (both peaks' peak+lower+upper rows) gave
    # window sums matching the two independently-run separate simulations to
    # within Monte Carlo noise (<1% on every window). So ONE simulation run
    # per segment, scoring every peak's windows at once, replaces N separate
    # full re-simulations of the SAME underlying photon transport physics --
    # close to an Nx reduction in simulation time for an N-peak isotope.
    # Monoenergetic builds (isotope=None) genuinely can't do this: each peak
    # would need its own single-energy source, so those keep one run per
    # peak, unchanged. Single-peak isotopes (Tc-99m, I-131) are unaffected
    # either way -- one peak is one run regardless.
    combine_windows = bool(isotope) and len(peaks) > 1
    win_prefix = (isd_name or "sim")       # e.g. lu177_113.win, i131_364.win
    win_rows_all = {}
    peak_win = {}
    window_offsets = {}   # "{center}kev" -> {"peak":, "lower":, "upper":} (1-based
                          # window numbers), only populated when combine_windows

    if combine_windows:
        # ONE file, every peak's rows back to back in peak order -- SIMIND
        # numbers scattwin output windows w1, w2, ... in the SAME order as
        # the .win file's rows, so this directly determines window_offsets.
        wfile = os.path.join(workdir, f"{win_prefix}_combined.win")
        peak_win_base = f"{win_prefix}_combined"
        rows_by_peak = {}
        w_idx = 1
        with open(wfile, "w") as f:
            for c, w in peaks:
                pk = int(c)
                lo, hi = c * (1 - w / 200), c * (1 + w / 200)
                f.write(f"{lo:.3f},{hi:.3f},0\n")
                rows = [{"role": "peak", "lo_keV": round(lo, 3), "hi_keV": round(hi, 3)}]
                offsets = {"peak": w_idx}
                w_idx += 1
                if tew_scatter:
                    tw = tew_width_pct
                    l_hi = lo; l_lo = lo - c * (tw / 100.0)
                    f.write(f"{l_lo:.3f},{l_hi:.3f},0\n")
                    rows.append({"role": "lower", "lo_keV": round(l_lo, 3),
                                 "hi_keV": round(l_hi, 3)})
                    offsets["lower"] = w_idx
                    w_idx += 1
                    u_lo = hi; u_hi = hi + c * (tw / 100.0)
                    f.write(f"{u_lo:.3f},{u_hi:.3f},0\n")
                    rows.append({"role": "upper", "lo_keV": round(u_lo, 3),
                                 "hi_keV": round(u_hi, 3)})
                    offsets["upper"] = w_idx
                    w_idx += 1
                rows_by_peak[pk] = rows
                window_offsets[f"{pk}kev"] = offsets
        win_rows_all = rows_by_peak
        # every peak in this build shares the one combined file/offsets
        for c, w in peaks:
            peak_win[int(c)] = peak_win_base
        log["resolved"]["win_files_per_peak"] = {str(k): os.path.abspath(wfile)
                                                  for k in peak_win}
    else:
        # legacy: one .win (and later one simind run) PER peak, windows
        # always 1/2/3 -- unchanged from before this feature existed.
        for c, w in peaks:
            pk = int(c)
            wfile = os.path.join(workdir, f"{win_prefix}_{pk}.win")
            peak_win[pk] = f"{win_prefix}_{pk}"    # base name for /fw (no extension)
            rows = []
            with open(wfile, "w") as f:
                lo, hi = c * (1 - w / 200), c * (1 + w / 200)
                f.write(f"{lo:.3f},{hi:.3f},0\n")               # w1 photopeak
                rows.append({"role": "peak", "lo_keV": round(lo, 3),
                             "hi_keV": round(hi, 3)})
                if tew_scatter:
                    tw = tew_width_pct
                    l_hi = lo; l_lo = lo - c * (tw / 100.0)
                    f.write(f"{l_lo:.3f},{l_hi:.3f},0\n")       # w2 lower scatter
                    rows.append({"role": "lower", "lo_keV": round(l_lo, 3),
                                 "hi_keV": round(l_hi, 3)})
                    u_lo = hi; u_hi = hi + c * (tw / 100.0)
                    f.write(f"{u_lo:.3f},{u_hi:.3f},0\n")       # w3 upper scatter
                    rows.append({"role": "upper", "lo_keV": round(u_lo, 3),
                                 "hi_keV": round(u_hi, 3)})
            win_rows_all[pk] = rows
        log["resolved"]["win_files_per_peak"] = {
            str(k): os.path.abspath(os.path.join(workdir, f"{win_prefix}_{k}.win"))
            for k in peak_win}

    # keep a combined <prefix>.win too (peak-only) for convenience/inspection
    win = os.path.join(workdir, f"{win_prefix}.win")
    with open(win, "w") as f:
        for c, w in peaks:
            lo, hi = c * (1 - w / 200), c * (1 + w / 200)
            f.write(f"{lo:.3f},{hi:.3f},0\n")
    log["resolved"]["win_rows_per_peak"] = win_rows_all
    log["resolved"]["combine_windows"] = combine_windows

    # reshape for BuildResult.windows: {peak_keV: {role: {lo_keV, hi_keV}}}
    windows = {
        pk: {row["role"]: {"lo_keV": row["lo_keV"], "hi_keV": row["hi_keV"]}
             for row in rows}
        for pk, rows in win_rows_all.items()
    }
    log["resolved"]["tew_scatter"] = tew_scatter

    commands, seg_info = [], []
    for si, (z0, z1) in enumerate(segs):
        tag = f"{pid}_{si:02d}"
        s_arr = src[z0:z1]
        d_arr = den[z0:z1]
        seg_nz = z1 - z0

        # Guard: a 512x512-class in-plane matrix is almost never what you want
        # for SPECT MC (this is exactly what caused SIMIND error 64 / EOF).
        if nx > 256 or ny > 256:
            raise ValueError(
                f"In-plane matrix {nx}x{ny} is too large for SPECT Monte Carlo. "
                f"Resample first with prepare_inputs(...) (e.g. 4 mm grid) so the "
                f"map is ~128x128. A 512x512 map triggers the 'error 64' EOF you saw.")

        # ---- axial sizing so slice count == axial_matrix --------------------
        # The detector has `axial_matrix` rows (index 77) and PyTomography
        # reconstructs an object that many voxels tall, with the SAME axial
        # voxel size as in-plane (`pix`). So each bed's axial extent is exactly
        # fov_axial_cm (the 40 cm scan range), independent of the in-plane
        # matrix. A full 40 cm bed already has axial_matrix slices, so no
        # padding is added; only a short trailing/partial segment (or a segment
        # slightly over due to rounding) is adjusted here. Any padding is air
        # (density 0, source 0), which doesn't change the physics, and the
        # attenuation map follows the phantom so amap and object stay the same
        # height (no reshape mismatch).
        z_pad_lo = 0
        if seg_nz < axial_matrix:
            total_pad = axial_matrix - seg_nz
            z_pad_lo = total_pad // 2
            z_pad_hi = total_pad - z_pad_lo
            pad_w = ((z_pad_lo, z_pad_hi), (0, 0), (0, 0))  # pad axis 0 (z)
            s_arr = np.pad(s_arr, pad_w, mode="constant", constant_values=0)
            d_arr = np.pad(d_arr, pad_w, mode="constant", constant_values=0)
            seg_nz = axial_matrix
        elif seg_nz > axial_matrix:
            # segment taller than the axial matrix: crop centrally (rare, from
            # rounding of the segment range vs fov_axial_cm/pix)
            excess = seg_nz - axial_matrix
            c0 = excess // 2
            s_arr = s_arr[c0:c0 + axial_matrix]
            d_arr = d_arr[c0:c0 + axial_matrix]
            z_pad_lo = -c0
            seg_nz = axial_matrix

        # density -> Integer*2 as 1000*density (index 14 = -1, *.dmi)
        dmi = np.clip(np.rint(d_arr * 1000.0), 0, 32767).astype("<i2")
        dmi_path = os.path.join(workdir, f"{tag}.dmi")
        dmi.tofile(dmi_path)

        # source -> Integer*2 counts (index 15 = -1, *.smi). Scale so the
        # brightest voxel emits `source_hist_max` histories; /NN multiplies
        # further. Lower source_hist_max => proportionally faster simulation
        # (runtime is linear in total histories). Relative activity between
        # voxels is preserved exactly regardless of this value.
        s_max = float(s_arr.max()) if s_arr.max() > 0 else 1.0
        smi = np.clip(np.rint(s_arr / s_max * source_hist_max),
                      0, 32767).astype("<i2")
        smi_path = os.path.join(workdir, f"{tag}.smi")
        smi.tofile(smi_path)

        # Hard consistency check: bytes written MUST equal idx34 * nx * ny * 2.
        # If this ever fails SIMIND raises error 64, so we catch it here first.
        expect_bytes = seg_nz * nx * ny * 2
        for pth in (dmi_path, smi_path):
            got = os.path.getsize(pth)
            if got != expect_bytes:
                raise AssertionError(
                    f"{pth}: wrote {got} bytes but SIMIND will expect "
                    f"idx34*nx*ny*2 = {seg_nz}*{nx}*{ny}*2 = {expect_bytes}. "
                    f"This mismatch is the cause of SIMIND error 64.")

        half_axial = (seg_nz * sz) / 2.0
        source_voxel_sum = int(smi.astype(np.int64).sum())

        # World origin of THIS segment's first slice. The prepared array is
        # (z,y,x); slicing src[z0:z1] shifts the origin along the z world axis
        # by z0 voxels. Axial padding (z_pad_lo slices added below the data)
        # moves the map's first slice further down, so the origin shifts by
        # -z_pad_lo voxels (padding added below) to keep the DATA at its true
        # world position.
        #
        # In-plane (x/y): unlike z, the .dmi/.smi files below are written at
        # the map's TRUE nx x ny extent (no np.pad here) -- but the .smc
        # header's declared image/projection matrix (index 76, see the
        # image-matrix-sizing block above) is matrix_size x matrix_size,
        # which is >= max(nx, ny) so the object is never truncated. Whenever
        # matrix_size > nx or > ny, SIMIND centers the smaller phantom matrix
        # within the larger declared detector matrix (same floor/ceil split
        # convention as the axial padding above), so the RECONSTRUCTED
        # object -- which comes out matrix_size x matrix_size -- has that
        # same centering baked in. Reusing the raw src_origin_mm for x/y (as
        # if no padding happened) put reconstructed volumes off by half the
        # padding on whichever in-plane axis needed it -- confirmed
        # empirically on a real build: a (114-93)//2 = 10-voxel / 48 mm
        # anterior-posterior shift between the reconstruction and the source
        # PET/CT, closed to <0.1 mm by this correction. pad_x_lo/pad_y_lo are
        # 0 on whichever axis already equals matrix_size (the common case for
        # the WIDER of the two in-plane axes, since matrix_size = max(nx,ny)).
        pad_x_lo = (matrix_size - nx) // 2
        pad_y_lo = (matrix_size - ny) // 2
        # direction9 is a row-major 3x3 unit-direction matrix; diagonal
        # entries (indices 0/4/8) give the +1/-1 sign relating increasing
        # array index to increasing world coordinate on that SAME axis. This
        # (like the pre-existing z-origin line below) assumes an axis-aligned
        # frame (no off-diagonal rotation/shear), true for this pipeline's
        # resampled PET/CT inputs.
        dir_x, dir_y, dir_z = src_direction[0], src_direction[4], src_direction[8]
        seg_origin_mm_xyz = [
            src_origin_mm[0] - dir_x * pad_x_lo * src_spacing_mm[0],
            src_origin_mm[1] - dir_y * pad_y_lo * src_spacing_mm[1],
            src_origin_mm[2] + dir_z * (z0 - z_pad_lo) * src_spacing_mm[2],
        ]

        # ---- DEBUG: save the maps SIMIND actually reads, as NIfTI ---------- #
        # These are the exact int16 arrays (post scaling/clipping) that go into
        # the .dmi/.smi, wrapped with this segment's world frame so they overlay
        # on the original PET/CT. Also dump the raw (pre-int16) density in
        # g/cm^3 and source in original units for comparison.
        debug_files = {}
        if save_debug_nifti:
            dbg = os.path.join(workdir, "debug_nifti")
            os.makedirs(dbg, exist_ok=True)
            frame = (seg_origin_mm_xyz, src_spacing_mm, src_direction)
            # what SIMIND reads (int16):
            debug_files["dmi_nifti"] = _map_to_nifti(
                dmi, os.path.join(dbg, f"{tag}_dmi_int16.nii.gz"), *frame)
            debug_files["smi_nifti"] = _map_to_nifti(
                smi, os.path.join(dbg, f"{tag}_smi_int16.nii.gz"), *frame)
            # human-facing physical values:
            debug_files["density_gcm3_nifti"] = _map_to_nifti(
                d_arr.astype("float32"),
                os.path.join(dbg, f"{tag}_density_gcm3.nii.gz"), *frame)
            debug_files["source_raw_nifti"] = _map_to_nifti(
                s_arr.astype("float32"),
                os.path.join(dbg, f"{tag}_source_raw.nii.gz"), *frame)

        # Real-data crop window within the (possibly padded/cropped) seg_nz
        # volume: the slices that came from actual phantom data, as opposed to
        # the air padding added above to reach axial_matrix. Recorded here so
        # downstream recon code can crop the pad out without re-deriving this
        # arithmetic (and getting the origin shift wrong).
        _real_nz = int(z1 - z0)
        if z_pad_lo >= 0:
            data_z0 = int(z_pad_lo)
            data_z1 = int(z_pad_lo + _real_nz)
        else:
            # segment was cropped (rare over-height case): no pad, full range
            data_z0 = 0
            data_z1 = int(seg_nz)
        data_z0 = max(0, min(data_z0, int(seg_nz)))
        data_z1 = max(data_z0, min(data_z1, int(seg_nz)))

        seg_record = {
            "tag": tag,
            "z_range": [int(z0), int(z1)],
            "n_slices": int(seg_nz),
            "z_pad_lo": int(z_pad_lo),
            "data_z_range": [data_z0, data_z1],
            "half_axial_cm": half_axial,
            "axial_cm": seg_nz * sz,
            # placement metadata: reconstructed bed voxels map to world coords
            # with this origin + the prepared spacing/direction above.
            "origin_mm_xyz": seg_origin_mm_xyz,
            "spacing_mm_xyz": src_spacing_mm,
            "direction": src_direction,
            "source_voxel_sum": source_voxel_sum,       # = histories/proj (pre /NN)
            "source_scale_smax": s_max,                 # pre-scale brightest voxel
            "hist_per_unit": source_hist_max / s_max,   # emitted hist per input unit
            "source_hist_max": source_hist_max,         # hist from brightest voxel
            "files": {
                "dmi": {"path": os.path.abspath(dmi_path),
                        "bytes": os.path.getsize(dmi_path),
                        "sha256": _sha256(dmi_path)},
                "smi": {"path": os.path.abspath(smi_path),
                        "bytes": os.path.getsize(smi_path),
                        "sha256": _sha256(smi_path)},
                "debug_nifti": {k: os.path.abspath(v)
                                for k, v in debug_files.items()},
            },
            "runs": [],     # one per photopeak (legacy) or one per segment (combined)
            # non-empty only when combine_windows: "{center}kev" -> {"peak":,
            # "lower":, "upper":} 1-based scattwin window numbers within this
            # segment's ONE combined simind run's output (same for every
            # segment in this build, since all segments share the same
            # peaks/tew_scatter). Empty dict -> legacy layout (see
            # resolve_peak_files, in the reconstruction section below).
            "window_offsets": dict(window_offsets),
        }

        if combine_windows:
            s = SMC(template_smc)
            center0, width0 = peaks[0]
            # Index 1/20/21 (energy + default %window) are only a fallback
            # SIMIND uses when NOT told otherwise -- the external .win file
            # (scattwin, /fw:/84:1) is what actually governs scoring, and
            # covers every peak regardless of what these say (empirically
            # confirmed, see the combine_windows comment above). So which
            # peak's center/width goes here is arbitrary; peaks[0] is just a
            # convenient, deterministic choice.
            s.title = f"{(isotope or 'src').title()} {'+'.join(str(int(c)) for c,_ in peaks)}keV {tag} {spec.name}".ljust(70)[:70]
            s.set(1, -center0)             # isotope routine (combine_windows implies isotope is set)
            s.set(8, spec.detector_x_cm)
            s.set(9, spec.crystal_thickness_cm)
            s.set(10, spec.detector_width_cm)
            s.set(12, radius_of_rotation_cm)
            s.set(22, spec.energy_res_pct)
            s.set(23, spec.intrinsic_fwhm_cm)
            s.set(20, -width0)
            s.set(21, -width0)
            s.set(14, -1)
            s.set(15, -1)
            s.set(31, voxel_cm)
            s.set(32, 0)
            s.set(33, 1)
            s.set(34, seg_nz)
            s.set(35, density_border_threshold)
            s.set(2, half_axial)
            s.set(5, half_axial)
            s.set(78, nx); s.set(79, nx)
            s.set(81, ny); s.set(82, ny)
            s.set(28, pix)
            s.set(29, n_azimuth)
            s.set(30, rotation_mode)
            s.set(76, matrix_size)
            s.set(77, axial_matrix)
            s.set_flag(1, on_line_printout)
            s.set_flag(4, True)
            s.set_flag(5, True)
            s.set_flag(7, True)
            s.set_flag(8, True)
            s.set_flag(10, True)
            s.set_flag(11, True)
            s.set_flag(2, False)
            s.set(84, 1)
            s.set_collimator(col_name)
            s.set_file(4, tag)
            s.set_file(5, tag)
            s.set_file(7, isd_name)

            base = tag    # NO peak suffix: one shared output for every peak
            smc_path = os.path.join(workdir, base + ".smc")
            s.write(smc_path)

            import shutil
            search_dirs = [d for d in [
                os.environ.get("SMC_DIR"), smc_dir, r"C:\simind\v9\smc_dir",
            ] if d]
            src_isd = None
            for d in search_dirs:
                cand = os.path.join(d, f"{isd_name}.isd")
                if os.path.exists(cand):
                    src_isd = cand
                    break
            if src_isd:
                dst = os.path.join(workdir, f"{isd_name}.isd")
                if os.path.abspath(src_isd) != os.path.abspath(dst):
                    shutil.copyfile(src_isd, dst)
                shutil.copyfile(src_isd, os.path.join(workdir, "none.isd"))
            else:
                print(f"  WARNING: {isd_name}.isd not found in {search_dirs}. "
                      f"Set smc_dir= or SMC_DIR, or copy {isd_name}.isd into "
                      f"the workdir manually. For {isotope}, the spectrum "
                      f"file must exist in your SIMIND isotope database.")

            nn = f"/NN:{photon_multiplier_nn}" if photon_multiplier_nn != 1 else ""
            switches = f"/fw:{peak_win_base} /84:1 /tr:14 /tr:15 /in:x22,5x"
            if nn:
                switches += f" {nn}"
            cmd = f"simind {base} {base} {switches}".rstrip()
            commands.append(cmd)

            seg_record["runs"].append({
                "combined_peaks_keV": [c for c, _ in peaks],
                "window_offsets": dict(window_offsets),
                "smc": {"path": os.path.abspath(smc_path),
                        "bytes": os.path.getsize(smc_path),
                        "sha256": _sha256(smc_path)},
                "command": cmd,
                "smc_written": s.provenance(),
            })

        else:
          # NOTE: unchanged legacy per-peak loop (indent kept as originally
          # written -- 2sp under `else:` then 4sp per level below -- to
          # avoid a risky whole-block reindent of working code).
          for (center, width) in peaks:
            s = SMC(template_smc)
            s.title = f"Lu177 {int(center)}keV {tag} {spec.name}".ljust(70)[:70]

            # geometry / detector
            # Index 1 = photon energy. POSITIVE => monoenergetic single line.
            # NEGATIVE => call the isotope routine and read the FULL emission
            # spectrum from an .isd file (real Lu-177: 113 + 208 keV + x-rays +
            # down-scatter). The absolute value still defines the energy window
            # centre. Real radionuclide physics (cross-talk, 208->113 down-
            # scatter) requires the negative/isotope path.
            if isotope:
                s.set(1, -center)          # isotope routine, window at |center|
            else:
                s.set(1, center)           # monoenergetic
            s.set(8, spec.detector_x_cm)
            s.set(9, spec.crystal_thickness_cm)
            s.set(10, spec.detector_width_cm)
            s.set(12, radius_of_rotation_cm)
            s.set(22, spec.energy_res_pct)
            s.set(23, spec.intrinsic_fwhm_cm)

            # energy windows: relative % (negative) centred on index 1
            s.set(20, -width)   # upper as % window
            s.set(21, -width)   # lower as % window

            # voxel phantom + source (Integer*2 maps)
            s.set(14, -1)                 # density map *.dmi
            s.set(15, -1)                 # source  map *.smi
            s.set(31, voxel_cm)           # voxel side [cm]
            s.set(32, 0)                  # transaxial slices (i,j)->(y,z)
            s.set(33, 1)                  # first image index (SIMIND is 1-based!
                                          # 0 => reads block -1 => error 64)
            s.set(34, seg_nz)             # number of density maps
            s.set(35, density_border_threshold)
            s.set(2, half_axial)          # source stack half axial length
            s.set(5, half_axial)          # phantom stack half axial length
            s.set(78, nx); s.set(79, nx)  # map matrix I (cols)
            s.set(81, ny); s.set(82, ny)  # map matrix J (rows)

            # acquisition
            s.set(28, pix)                # image pixel size
            s.set(29, n_azimuth)          # projections
            s.set(30, rotation_mode)      # rotation
            s.set(76, matrix_size)      # image cols (I): in-plane
            s.set(77, axial_matrix)     # image rows (J): axial = scan range

            # ---- simulation flags ------------------------------------------
            # Flag 5 = SPECT: rotate through idx29 projections and write the
            # *.a00 stack (Real*4). WITHOUT this SIMIND makes ONE planar frame.
            s.set_flag(1, on_line_printout)   # on-line printout (see param docstring)
            s.set_flag(4, True)    # simulate the collimator (essential)
            s.set_flag(5, True)    # <-- SPECT mode: all projections
            s.set_flag(7, True)    # backscatter / PMT layer
            s.set_flag(8, True)    # random seed varies between runs
            s.set_flag(10, True)   # simulate protective cover
            s.set_flag(11, True)   # interactions in voxel phantom
            # Flag 2 (planar .bim) is forced FALSE by SIMIND when Flag 5 is set.
            s.set_flag(2, False)

            # scoring: scattwin (uses the .win file)
            s.set(84, 1)

            s.set_collimator(col_name)
            # main-menu file slots: 5th=source base, 6th=density base
            s.set_file(4, tag)            # source map base name
            s.set_file(5, tag)            # density map base name
            # File slot 7 = main-menu item 14 ("Energy resolution file"). When
            # Index 1 is NEGATIVE, SIMIND reads THIS field as the isotope
            # spectrum base name and loads <name>.isd. Leaving it 'none' makes
            # SIMIND look for none.isd. So set it to the isotope name; SIMIND
            # then loads <isotope>.isd from the working dir / SMC_DIR.
            if isotope:
                s.set_file(7, isd_name)   # -> loads e.g. lu177.isd

            base = f"{tag}_{int(center)}kev"
            smc_path = os.path.join(workdir, base + ".smc")
            s.write(smc_path)

            # SIMIND reads file slot 7 (menu 14) as the isotope spectrum base
            # name and loads <name>.isd from the working directory. We set slot
            # 7 = isotope above, so SIMIND looks for <isotope>.isd. Copy that
            # file into the workdir under its ISOTOPE name (not the input base
            # name) so the field-14 lookup finds it.
            if isotope:
                import shutil
                search_dirs = [d for d in [
                    os.environ.get("SMC_DIR"), smc_dir,
                    r"C:\simind\v9\smc_dir",
                ] if d]
                src_isd = None
                for d in search_dirs:
                    cand = os.path.join(d, f"{isd_name}.isd")
                    if os.path.exists(cand):
                        src_isd = cand
                        break
                if src_isd:
                    # workdir copy under the isotope name (matches field 14)
                    dst = os.path.join(workdir, f"{isd_name}.isd")
                    if os.path.abspath(src_isd) != os.path.abspath(dst):
                        shutil.copyfile(src_isd, dst)
                    # GUARANTEED fallback: this SIMIND build looks for none.isd
                    # in the working dir regardless of field 14. Copying the
                    # spectrum to none.isd makes the isotope routine find it.
                    # (Confirmed empirically: a file named none.isd lets it run.)
                    shutil.copyfile(src_isd,
                                    os.path.join(workdir, "none.isd"))
                else:
                    print(f"  WARNING: {isd_name}.isd not found in {search_dirs}. "
                          f"Set smc_dir= or SMC_DIR, or copy {isd_name}.isd into "
                          f"the workdir manually. For {isotope}, the spectrum "
                          f"file must exist in your SIMIND isotope database.")

            nn = f"/NN:{photon_multiplier_nn}" if photon_multiplier_nn != 1 else ""
            # SIMIND <input> <output> /switches
            #   /fw:lu177  -> scattwin reads the lu177.win energy windows
            #   /84:1      -> scattwin scoring routine
            #   /tr:14     -> write Interfile headers (.h00) for PyTomography
            #   /tr:15     -> write the aligned attenuation map (.hct/.ict)
            #   /in:x22,5x -> write the attenuation map as mu (1/cm), float
            switches = f"/fw:{peak_win[int(center)]} /84:1 /tr:14 /tr:15 /in:x22,5x"
            # isotope spectrum is set via file slot 7 in the .smc (menu 14),
            # NOT via /fi or /if — those don't override the field on this build.
            if nn:
                switches += f" {nn}"
            cmd = f"simind {base} {base} {switches}".rstrip()
            commands.append(cmd)

            # per-run provenance: exactly what went into this .smc
            run_rec = {
                "peak_center_keV": center,
                "peak_width_pct": width,
                "smc": {"path": os.path.abspath(smc_path),
                        "bytes": os.path.getsize(smc_path),
                        "sha256": _sha256(smc_path)},
                "command": cmd,
                "smc_written": s.provenance(),
            }
            seg_record["runs"].append(run_rec)

        seg_info.append(dict(tag=tag, z=(z0, z1), n_slices=seg_nz,
                             dmi=dmi_path, smi=smi_path,
                             axial_cm=seg_nz * sz))
        log["segments"].append(seg_record)

    run = os.path.join(workdir, "run_all.sh")
    with open(run, "w", newline="\n") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        if simind_bin_dir:
            # prepended so this script finds `simind` on its own -- no need
            # to `export PATH=...` by hand before running it, and no need to
            # `cd` into this folder first (see the `cd` line right below,
            # which makes the script find its OWN .smc/.win files wherever
            # it's invoked from, e.g. `bash /abs/path/to/run_all.sh`).
            f.write(f'export PATH="{simind_bin_dir}:$PATH"\n')
        f.write('cd "$(dirname "$0")"\n')
        f.write("# scattwin reads lu177.win via /fw:lu177\n")
        f.write("\n".join(commands) + "\n")
    os.chmod(run, 0o755)

    # Windows batch equivalent (cmd.exe). CRLF line endings.
    run_bat = os.path.join(workdir, "run_all.bat")
    with open(run_bat, "w", newline="\r\n") as f:
        f.write("@echo off\r\n")
        f.write("REM run from this folder so SIMIND finds the .smc/.win files\r\n")
        f.write('cd /d "%~dp0"\r\n')
        f.write("REM scattwin reads lu177.win via /fw:lu177\r\n")
        for c in commands:
            f.write(c + "\r\n")

    # Windows MPI batch: load Intel MPI env, then run each case with mpiexec.
    run_mpi = os.path.join(workdir, "run_all_mpi.bat")
    intel_vars = (r"C:\Program Files (x86)\Intel\oneAPI\mpi\latest\env\vars.bat")
    with open(run_mpi, "w", newline="\r\n") as f:
        f.write("@echo off\r\n")
        f.write('cd /d "%~dp0"\r\n')
        f.write("REM load Intel MPI runtime (mpiexec + impi.dll on PATH)\r\n")
        f.write(f'call "{intel_vars}"\r\n')
        f.write(f"REM {mpi_ranks} ranks; simind_mpi replaces simind\r\n")
        for c in commands:
            # turn "simind <args>" into "mpiexec -n N simind_mpi <args>"
            mpi_c = c.replace("simind ", f"mpiexec -n {mpi_ranks} simind_mpi ", 1)
            f.write(mpi_c + "\r\n")

    log["outputs"] = {
        "run_script": os.path.abspath(run),
        "run_script_bat": os.path.abspath(run_bat),
        "run_script_mpi_bat": os.path.abspath(run_mpi),
        "commands": commands,
        "n_smc_files": sum(len(s["runs"]) for s in log["segments"]),
    }
    log["finished"] = datetime.datetime.now().astimezone().isoformat()

    # ---- write the JSON provenance log ------------------------------------ #
    # Fixed name (not timestamped): rebuilding the same workdir overwrites
    # it, which matches how it's already used -- load_build_log() /
    # find_build_dirs() always want the LATEST build for that folder anyway.
    if log_file is None:
        log_file = os.path.join(workdir, "simulation-debug.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return BuildResult(workdir, seg_info, commands, win, log_file, windows)


# ============================================================================
#  SECTION 2 — RECONSTRUCTION  (was reconstruct_lu177_multibed.py)
# ============================================================================
"""
Reconstruct Lu-177 multi-bed SIMIND SPECT with attenuation correction
(and optional TEW scatter correction), then stitch beds axially — and write
SPATIALLY-REGISTERED output (NIfTI) so each bed and the stitched volume land
in the original patient world frame, independent of the raw .npy.

Built on the PyTomography SIMIND tutorial API:
  https://pytomography.readthedocs.io/en/latest/notebooks/t_siminddata.html

Placement metadata comes from the builder's JSON log (build_log_*.json):
  inputs.prepared_source_{origin,spacing,direction}   -> world frame
  segments[i].origin_mm_xyz                            -> per-bed world origin
So reconstructions can be overlaid on the original PET/CT in any viewer
(3D Slicer, ITK-SNAP, etc.) with no manual alignment.

PREREQUISITE — SIMIND runs produced Interfile headers (Flag 14 /tr:14) and an
aligned attenuation map (Flag 15 /tr:15). Ideally also /in:x22,5x so the .hct
is already in 1/cm (then set CONVERT_DENSITY_TO_MU=False).
"""

# ----------------------------------------------------- interfile / diagnostics
def read_interfile_keys(h00):
    d = {}
    for line in open(h00, "r", errors="ignore"):
        if ":=" in line:
            k, v = line.split(":=", 1)
            d[k.strip().lstrip("!;# ").strip()] = v.strip()
    return d


def radius_distance_unit(h00):
    """PyTomography's simind.get_metadata() needs to be told whether the
    header's 'Radius' field is in mm or cm — SIMIND itself doesn't say, and
    different SIMIND versions disagree (confirmed: v8.0 writes cm, v9.0
    writes mm; get_metadata defaults to 'cm' if not told otherwise). Getting
    this wrong doesn't crash anything — it silently feeds a ~10x-wrong
    object-to-collimator distance into the PSF's distance-dependent blur
    formula, producing a massively oversized kernel that smears the whole
    reconstruction into one indistinct blob regardless of counts, views, or
    iterations (confirmed on the NEMA phantom: identical blob at 16 and 128
    views, since it's a deterministic geometry bug, not a statistics one).

    Rather than branch on the version string (fragile — we only know 2 data
    points), use physical plausibility: a SPECT radius of rotation for any
    human/phantom scan is essentially always 10-60 cm. A raw header value
    that large only makes sense as mm; anything already in that range is cm.
    """
    k = read_interfile_keys(h00)
    raw = float(k.get("Radius", "25"))
    return "mm" if raw > 120 else "cm"


def print_diagnostics(h00):
    """Surface the values that most often explain bad recon: collimator
    geometry (huge holes/thickness = blur), radius, matrix, energy window."""
    k = read_interfile_keys(h00)
    print("  --- projection header diagnostics ---")
    for key in ["matrix size [1]", "matrix size [2]", "number of projections",
                "extent of rotation", "direction of rotation",
                "scaling factor (mm/pixel) [1]", "Radius",
                "energy window lower level", "energy window upper level",
                "Collimator"]:
        if key in k:
            print(f"    {key:<32}: {k[key]}")
    # NOTE: collimator hole/thickness header fields are printed in unreliable
    # units by SIMIND, so they are intentionally NOT shown. Verify the
    # collimator against collim.col instead (SI-MELP = 0.294 cm hole,
    # 4.064 cm thick). The simulation used the correct collim.col geometry.


def orient_projection(proj_tensor):
    """Apply the per-frame orientation correction for SIMIND projections so
    both the reconstruction AND the saved NIfTI use the correct view.

    SIMIND writes each projection frame transposed/flipped relative to the
    expected detector orientation. The correction, verified by the user in
    SimpleITK, is:  PermuteAxes([1,0,2]) then Flip([True,False,False]) on an
    image whose axis order is (u, v, angle).

    We apply that EXACT SimpleITK op (not a hand-rolled numpy guess) by
    converting torch -> sitk -> corrected -> torch, so the operation is
    provably identical to the one the user confirmed looks correct.

    proj_tensor: torch (angle, u, v)  [PyTomography/SIMIND convention]
    returns:     torch (angle, u, v)  corrected
    """
    import torch as _torch
    arr = proj_tensor.detach().cpu().numpy().astype("float32")   # (angle,u,v)
    # sitk GetImageFromArray reads (z,y,x); to get an IMAGE with axis order
    # (x=u, y=v, z=angle) we must feed an array shaped (angle, v, u).
    arr_zyx = np.ascontiguousarray(np.transpose(arr, (0, 2, 1)))  # (angle,v,u)
    img = sitk.GetImageFromArray(arr_zyx)                         # image (u,v,angle)
    out = sitk.Flip(sitk.PermuteAxes(img, [1, 0, 2]), [True, False, False])
    # back to array (z,y,x) of the corrected image, then to torch (angle,u,v)
    out_arr = sitk.GetArrayFromImage(out)                        # (angle, v', u')
    # after permute+flip the image axis order is (v,u,angle) -> array (angle,u,v)
    corrected = np.ascontiguousarray(out_arr)                    # (angle,u,v)
    return _torch.as_tensor(corrected, dtype=proj_tensor.dtype,
                            device=proj_tensor.device)


def projections_to_nifti(proj_tensor, h00, out_path):
    """Save a projection stack as NIfTI for INSPECTION. Applies the same
    orientation correction as used for reconstruction, so what you see here is
    what PyTomography reconstructs from. Detector-frame, not patient space."""
    corrected = orient_projection(proj_tensor)
    arr = np.ascontiguousarray(corrected.detach().cpu().numpy().astype("float32"))
    # write in the corrected (angle,u,v) order; array->image gives (v,u,angle)
    img = sitk.GetImageFromArray(np.transpose(arr, (0, 2, 1)))   # (angle,v,u)->image(u,v,angle)
    k = read_interfile_keys(h00)
    du = float(k.get("scaling factor (mm/pixel) [1]", "4.0"))
    dv = float(k.get("scaling factor (mm/pixel) [2]", "4.0"))
    # angle axis "voxel size" = angular step between projections (deg), so all
    # 3 dims carry a real physical extent instead of a placeholder 1.0.
    extent = float(k.get("extent of rotation", "360"))
    nproj = int(float(k.get("number of projections", arr.shape[0])))
    dangle = extent / nproj if nproj else 1.0
    img.SetSpacing([du, dv, dangle])                # (u, v, angle[deg])
    img.SetOrigin([0.0, 0.0, 0.0])
    sitk.WriteImage(img, out_path)
    return img

# ----------------------------------------------------------------- settings
WORKDIR = r"C:\simind\v9\patients\NEMA\siemens_symbia_t--16--views"
PEAK = "113kev"            # or "208kev"; run each peak separately
BEDS = None # ["seg00"]   # auto-detected from the build log's segment tags, in build
              # order ("{patient_id}_00", "{patient_id}_01", ...). Only set
              # this to a literal list to override auto-detection or run a
              # subset of beds.

# Path to the builder's JSON log. Default: newest build_log_*.json in WORKDIR.
BUILD_LOG = None           # or set an explicit path

ACTIVITY_MBQ = 7.5e+6       # activity (MBq) used for the Poisson realization
                            # -- how this is applied across beds is set by
                            # ACTIVITY_MBQ_MODE below.
# "per_bed"       (default): the SAME ACTIVITY_MBQ is applied to every bed,
#                  only corrected for their RELATIVE brightness
#                  (segment_activity_scale) -- ACTIVITY_MBQ is a per-bed
#                  acquisition-statistics knob, not a whole-study total.
# "total_by_counts": ACTIVITY_MBQ is treated as the TOTAL activity across
#                  the WHOLE multi-bed study, and split across beds by each
#                  bed's share of total simulated source counts
#                  (segment_count_fraction) -- e.g. a bed holding 20% of the
#                  patient's total counts gets 0.2 * ACTIVITY_MBQ.
ACTIVITY_MBQ_MODE = "per_bed"
TIME_PER_PROJ_S = 1.0     # acquisition time per projection (s)
N_ITERS = 16
N_SUBSETS = 8

# ---- reconstruction algorithm -----------------------------------------
# Every algorithm below is a genuine, working option (verified against the
# installed pytomography==3.3.2's real class signatures, not just tutorial
# text) -- see https://pytomography.readthedocs.io/en/stable/notebooks/t_algorithms.html
#   "OSEM"    : ordered-subsets EM (default; what this pipeline always used).
#   "MLEM"    : plain EM, no subsets (n_subsets is ignored for this one).
#   "OSMAPOSL": OSEM-style MAP with a prior -- REQUIRES PRIOR_BETA.
#   "BSREM"   : block-sequential regularized EM with a prior -- REQUIRES
#               PRIOR_BETA. RELAXATION_SEQUENCE optional (pytomography has
#               its own sensible default if left None).
#   "RBIEM"   : rescaled block-iterative EM; prior optional (PRIOR_BETA may
#               be left None).
#   "RBIMAP"  : RBI + a prior -- REQUIRES PRIOR_BETA.
#   "SART"    : simultaneous algebraic reconstruction; no prior, has its own
#               built-in relaxation schedule.
# All 7 above were verified to actually run correctly against the installed
# pytomography==3.3.2 (real end-to-end test, not just import checks).
# NOT available:
#   "FBP" (filtered backprojection) -- pytomography==3.3.2's
#   FilteredBackProjection is confirmed BROKEN (a missing attribute
#   assignment, then a len()-on-an-int bug) — a library bug, not fixable
#   from here without monkey-patching internals we can't verify are then
#   mathematically correct. Selecting it raises NotImplementedError.
# NOT wired up (need inputs this pipeline doesn't have): KEM (needs a
# support/kernel image), DIPRecon (needs a deep-image-prior network),
# PGAAMultiBedSPECT (handles multi-bed itself, would conflict with this
# module's own bed-stitching).
ALGORITHM = "OSEM"
# Prior for OSMAPOSL/BSREM/RBIMAP (and optionally RBIEM): "relative_difference"
# (RelativeDifferencePrior, uses PRIOR_GAMMA too) or "quadratic" (QuadraticPrior,
# PRIOR_GAMMA unused). PRIOR_BETA=None -> no prior; required for OSMAPOSL/
# BSREM/RBIMAP, optional for RBIEM, unused for OSEM/MLEM/SART/FBP.
PRIOR_TYPE = "relative_difference"
PRIOR_BETA = None
PRIOR_GAMMA = 1.0
RELAXATION_SEQUENCE = None   # BSREM only; None -> pytomography's own default

USE_TEW = True             # per-peak win files now include w2/w3 scatter windows

# ---- projection / recon orientation ---------------------------------------
# SIMIND writes each projection frame transposed+flipped. Two independent
# knobs, because they touch different outputs:
#
#   ORIENT_PROJECTIONS_FOR_RECON:
#       If True, the SAME correction you verified in SimpleITK
#       (PermuteAxes([1,0,2]) + Flip([True,False,False])) is applied to the
#       projections BEFORE they enter OSEM, so PyTomography reconstructs from
#       the correctly-oriented view. Turn this on if your recon comes out
#       viewed from the wrong angle / mirrored.
#       CAUTION: if you enable this, the reconstructed volume orientation
#       changes too — you may then NOT need the axial flip in to_sitk. If the
#       recon ends up double-flipped, set FLIP_RECON_AXIAL=False.
#
#   FLIP_RECON_AXIAL (passed to to_sitk):
#       Independent z-flip of the reconstructed VOLUME to match PET/CT, used to
#       undo SIMIND's axial phantom flip. Leave True unless the recon comes out
#       upside-down after the projection correction.
#
# The saved projection NIfTI (proj_*.nii.gz) ALWAYS gets the verified
# correction so what you inspect equals what PyTomography sees when
# ORIENT_PROJECTIONS_FOR_RECON is True.
ORIENT_PROJECTIONS_FOR_RECON = True
FLIP_RECON_AXIAL = False

# .hct units. If you ran SIMIND with /in:x22,5x the map is already 1/cm ->
# set CONVERT_DENSITY_TO_MU=False. Otherwise .hct is g/dm3 (=density*1000).
CONVERT_DENSITY_TO_MU = True
# cm^2/g, water, NIST XCOM (https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html).
# 113/208 keV (Lu-177) are the original values already used throughout this
# pipeline -- the NEAREST tabulated NIST grid point (100/200 keV), not
# interpolated to the exact peak energy (0.171/0.137 match NIST's 100/200 keV
# rows exactly). Left as-is rather than changed underneath existing results.
# Every other entry below IS log-log interpolated to its exact peak energy
# (the standard method for attenuation coefficients, since they're smooth in
# log-log space) from the same NIST table -- covers every peak
# photopeak_cheatsheet lists, so peak="auto" works for every isotope in
# ISOTOPES, not just Lu-177. Add an entry here (or pass mu_water= directly)
# for any isotope/peak not already covered.
MU_WATER_TABLE = {
    "113kev": 0.171,    # Lu-177 (nearest-grid-point, see above)
    "208kev": 0.137,    # Lu-177 (nearest-grid-point, see above)
    "140kev": 0.154,    # Tc-99m 140.5 keV (log-log interpolated)
    "364kev": 0.110,    # I-131 364.5 keV (log-log interpolated)
    "218kev": 0.133,    # Ac-225 daughter Fr-221 218 keV (log-log interpolated)
    "440kev": 0.102,    # Ac-225 daughter Bi-213 440 keV (log-log interpolated)
}
MU_WATER = MU_WATER_TABLE[PEAK]                       # cm^2/g, water (NIST)
DENSITY_TO_MU_FACTOR = MU_WATER / 1000.0              # raw g/dm3 -> 1/cm


# --------------------------------------------------------------- build log
def load_build_log():
    """simulation-debug.json is the current fixed name the builder writes
    (see simind_lu177_v9.py); build_log_*.json is the older timestamped name
    from before that change, kept here so folders built before the rename
    still reconstruct fine."""
    path = BUILD_LOG
    if path is None:
        fixed = os.path.join(WORKDIR, "simulation-debug.json")
        if os.path.exists(fixed):
            path = fixed
        else:
            cands = sorted(glob.glob(os.path.join(WORKDIR, "build_log_*.json")))
            if not cands:
                print("WARNING: no simulation-debug.json / build_log_*.json "
                      "found; output will use the SPECT header frame only "
                      "(origin defaults to 0).")
                return None
            path = cands[-1]
    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)
    print(f"using build log: {path}")
    return log


def resolve_peak(peak, log=None, tol_kev=0.5):
    """PEAK accepts either a bare tag string (e.g. "113kev", matching the
    filenames the builder actually wrote) or an explicit window spec dict
    {"center_kev": 133, "lower_pct": 5, "upper_pct": 5} — a more readable
    way to say which peak you mean than remembering the tag string.

    IMPORTANT: SIMIND's window WIDTH is fixed at simulation time (the
    filename only ever encodes the center, e.g. "133kev" regardless of what
    width was simulated) — reconstruction cannot re-window data after the
    fact. So a dict spec here does NOT change what gets reconstructed; it's
    converted to the same "{int(center_kev)}kev" tag the builder already
    uses, and — if `log` is given — the requested lower_pct/upper_pct are
    checked against what was ACTUALLY simulated for that peak (from the
    build log's resolved.win_rows_per_peak), raising a clear error if they
    don't match within tol_kev. That way a mismatched request fails loudly
    instead of silently reconstructing the wrong window. To actually
    reconstruct a NEW window, rebuild with a matching
    energy_windows=[(center_kev, width_pct)] first (see
    simind_lu177_v9.build_simind_lu177).
    """
    if isinstance(peak, str):
        return peak
    if not isinstance(peak, dict) or "center_kev" not in peak:
        raise ValueError(
            'peak must be a string like "113kev" (matching a simulated '
            'build\'s filenames) or a dict {"center_kev":, "lower_pct":, '
            '"upper_pct":}')
    center = float(peak["center_kev"])
    default_half_width = peak.get("width_pct", 10.0) / 2.0
    lower_pct = float(peak.get("lower_pct", default_half_width))
    upper_pct = float(peak.get("upper_pct", default_half_width))
    tag = f"{int(center)}kev"

    if log is not None:
        rows = log.get("resolved", {}).get("win_rows_per_peak", {})
        peak_row = None
        for k, v in rows.items():
            if int(float(k)) == int(center):
                peak_row = next((r for r in v if r.get("role") == "peak"), None)
                break
        if peak_row is None:
            raise ValueError(
                f"No simulated peak found at {center:g} keV in this build's "
                f"log — available peaks: {list(rows)}. Rebuild with "
                f"energy_windows=[({center:g}, ...)] first.")
        want_lo = center * (1 - lower_pct / 100.0)
        want_hi = center * (1 + upper_pct / 100.0)
        got_lo, got_hi = peak_row["lo_keV"], peak_row["hi_keV"]
        if abs(want_lo - got_lo) > tol_kev or abs(want_hi - got_hi) > tol_kev:
            raise ValueError(
                f"Requested window {center:g} keV +{upper_pct:g}%/-{lower_pct:g}% "
                f"= [{want_lo:.2f}, {want_hi:.2f}] keV, but the simulated "
                f"peak at {center:g} keV was actually windowed to "
                f"[{got_lo:.2f}, {got_hi:.2f}] keV. Rebuild with a matching "
                f"energy_windows=[({center:g}, ...)] to get this exact window.")
    return tag


def _find_segment(log, bed):
    """Look up a segment by tag, case-INSENSITIVE. SIMIND itself lowercases
    every file it writes (tot_w1.h00, .hct, .ict, ...) regardless of the tag's
    original case, while our .dmi/.smi/.smc/log keep the tag as given (e.g.
    "NEMA_00" from patient_id="NEMA"). Windows' filesystem hides this — file
    access works either way — but a plain "==" tag lookup does not, and
    silently returns no match (None) instead of erroring, so a bed whose tag
    case doesn't match the log exactly loses its frame AND its pad crop with
    no warning. Comparing case-insensitively makes BEDS robust regardless of
    what case it ends up in (auto-detected from the log, or hand-typed)."""
    if log is None or not bed:
        return None
    bed_l = bed.lower()
    for seg in log.get("segments", []):
        if str(seg.get("tag", "")).lower() == bed_l:
            return seg
    return None


def segment_frame(log, bed):
    """Return (origin_mm_xyz, spacing_mm_xyz, direction9) for a bed from the
    build log, or None if unavailable."""
    seg = _find_segment(log, bed)
    if seg is None:
        return None
    return (seg.get("origin_mm_xyz"), seg.get("spacing_mm_xyz"),
            seg.get("direction"))


def auto_beds(log):
    """Bed tags in build order, straight from the build log's segments list
    (each tagged "{patient_id}_00", "{patient_id}_01", ... by the builder).
    Lets the recon script run off just WORKDIR + PEAK, no hand-copied BEDS."""
    if log is None:
        raise RuntimeError(
            "BEDS=None (auto) but no build_log_*.json was found in WORKDIR — "
            "auto-detection needs it to know the segment tags. Either point "
            "WORKDIR at a folder containing the build log, or set BEDS "
            "explicitly to the tag list printed by the builder.")
    tags = [seg.get("tag") for seg in log.get("segments", [])]
    if not tags:
        raise RuntimeError("build log has no segments; set BEDS explicitly.")
    return tags


def segment_activity_scale(log, bed):
    """Per-bed correction for SIMIND's independent source normalization.

    The builder scales EACH segment's .smi so THAT SEGMENT's own brightest
    voxel emits `source_hist_max` histories (see simind_lu177_v9.py, s_max is
    computed per-segment). Two beds with different true relative activity
    therefore both get simulated with their local peak hitting the SAME
    history count, which erases the real brightness relationship between
    beds — e.g. a genuinely dim bed and a genuinely bright bed can come back
    from SIMIND looking equally bright. Recon-side, this shows up as beds
    with visibly mismatched dark/bright levels that aren't in the source.

    The log records, per segment, source_scale_smax (that segment's local
    s_max, in whatever arbitrary units the source NIfTI uses). The RATIO of
    source_scale_smax between two segments in the same log is exactly the
    ratio of their true relative peak activity, regardless of what constant
    it's divided by — so normalizing against `source_hist_max` (a fixed
    builder constant, NOT a physical calibration) preserves that ratio too,
    but it also rescales the absolute magnitude by however small the source
    array's raw units happen to be (e.g. peak=20 out of source_hist_max=1000
    crushes every bed's statistics 50x, starving the Poisson realization —
    confirmed on the NEMA phantom, "big cloud" recon instead of the spheres).

    Instead normalize against the BRIGHTEST segment in this same log: that
    bed gets scale=1.0 (full, uncorrected magnitude — exactly what a
    single-bed case always gets, since it IS the brightest of one), and every
    other bed is scaled down proportionally to its true relative activity.
    This keeps the cross-bed relative-brightness fix (ratios are identical
    either way) while no longer touching absolute count statistics for the
    common case of one bed, or for whichever bed is already the brightest.

    Returns 1.0 (no-op) if the log or fields are unavailable (older logs).
    """
    if log is None:
        return 1.0
    seg = _find_segment(log, bed)
    if seg is None:
        return 1.0
    s_max = seg.get("source_scale_smax")
    if not s_max:
        return 1.0
    all_smax = [s.get("source_scale_smax") for s in log.get("segments", [])
                if s.get("source_scale_smax")]
    if not all_smax:
        return 1.0
    ref = max(all_smax)
    if ref <= 0:
        return 1.0
    return float(s_max) / float(ref)


def _segment_raw_count(seg):
    """This segment's total source counts in RAW (pre-.smi-scaling) units:
    un-scale source_voxel_sum (sum of the .smi values SIMIND actually
    simulated) back through that segment's own s_max/source_hist_max boost
    (the same per-segment conversion segment_activity_scale uses for a
    single voxel's value, applied here to the whole-segment sum instead).
    This is necessary because each segment was independently scaled by a
    DIFFERENT factor at build time (see simind_lu177_v9.py) — comparing raw
    source_voxel_sum across segments directly would be comparing numbers on
    different scales. Returns None if any needed field is missing/zero."""
    voxel_sum = seg.get("source_voxel_sum")
    s_max = seg.get("source_scale_smax")
    hist_max = seg.get("source_hist_max")
    if not voxel_sum or not s_max or not hist_max:
        return None
    return float(voxel_sum) * float(s_max) / float(hist_max)


def _compute_true_activity_fractions(log):
    """Each segment's TRUE (non-duplicated) share of total body activity,
    read directly from the original whole-body prepared source NIfTI
    (log["arguments"]["source_nifti"]) -- NOT from summing each segment's
    own .smi-derived count (_segment_raw_count), which DOUBLE-COUNTS every
    bed's overlap with its neighbours: adjacent segments' z-ranges
    intentionally overlap by fov_overlap_cm (see simind_lu177_v9's
    _segments()), so summing "this segment's own total" across every
    segment counts each overlapping strip once per bed sharing it --
    confirmed empirically (2026-08-12, psma_000000): bed0 came out 25-33%
    brighter than bed1 at the SAME physical location in their shared
    overlap, comparing their independent reconstructions directly.

    Fix: read the whole-body source array ONCE (the same file every
    segment's own z0:z1 window -- recorded per-segment as "z_range" -- was
    carved from), and split each z-slice's true activity EXACTLY once
    across whichever segment(s) claim it: a slice inside only one
    segment's range counts fully (1x) toward that segment; a slice shared
    by two ADJACENT segments' overlap counts HALF (0.5x) toward each. This
    is only exact for pairwise (2-way) overlaps between immediate
    neighbours -- the normal case for this pipeline's segment geometry
    (fov_axial_cm segments each much taller than 2x fov_overlap_cm); a
    3-way overlap would need a different split, but shouldn't occur under
    normal build settings. Summing every segment's resulting share this
    way reproduces the TRUE whole-body total exactly once, so fractions
    across beds sum to 1.0 -- matching activity_mbq_mode="total_by_counts"
    's documented "splits the WHOLE study's total activity across beds"
    semantics.

    Returns {lowercased_tag: fraction}, or {} if the source NIfTI can't be
    found/read (caller falls back to the old .smi-derived approximation).
    """
    src_path = log.get("arguments", {}).get("source_nifti")
    if not src_path or not os.path.exists(src_path):
        return {}
    try:
        src_img = sitk.ReadImage(src_path)
    except Exception:
        return {}
    src = sitk.GetArrayFromImage(src_img).astype(np.float64)   # (z,y,x)
    total_body_activity = float(src.sum())
    if total_body_activity <= 0:
        return {}
    per_slice = src.reshape(src.shape[0], -1).sum(axis=1)      # (nz,) one pass

    segs = log.get("segments", [])
    # sort by z_range start defensively (same reasoning as stitch_by_origin
    # -- don't assume the log's own segment order is already sorted)
    segs_sorted = sorted(
        (s for s in segs if s.get("z_range") and s.get("tag") is not None),
        key=lambda s: s["z_range"][0])

    fractions = {}
    n = len(segs_sorted)
    for i, seg in enumerate(segs_sorted):
        z0, z1 = int(seg["z_range"][0]), int(seg["z_range"][1])
        if z1 <= z0:
            continue
        weight = np.ones(z1 - z0, dtype=np.float64)
        if i > 0:
            # halve the slices shared with the PREVIOUS segment (this
            # segment's own start, up to where the previous one ends)
            prev_z1 = int(segs_sorted[i - 1]["z_range"][1])
            overlap_end = min(prev_z1, z1)
            if overlap_end > z0:
                weight[0:overlap_end - z0] *= 0.5
        if i < n - 1:
            # halve the slices shared with the NEXT segment (from where the
            # next one starts, up to this segment's own end)
            next_z0 = int(segs_sorted[i + 1]["z_range"][0])
            overlap_start = max(next_z0, z0)
            if overlap_start < z1:
                weight[overlap_start - z0:z1 - z0] *= 0.5
        this_bed_activity = float((per_slice[z0:z1] * weight).sum())
        fractions[str(seg["tag"]).lower()] = this_bed_activity / total_body_activity
    return fractions


def segment_count_fraction(log, bed):
    """This bed's share of the TOTAL source counts across every bed in this
    log (the whole multi-bed study), as a fraction that sums to 1.0 over all
    beds — e.g. a bed holding 20% of the patient's total simulated counts
    returns 0.2. Use this (via reconstruct's activity_mbq_mode=
    "total_by_counts") when ACTIVITY_MBQ is meant to represent the WHOLE
    study's total activity, to be split across beds by their share of
    counts, instead of the default "per_bed" mode where the SAME
    ACTIVITY_MBQ is applied to every bed and only relative brightness is
    corrected (segment_activity_scale).

    Preferentially computed by _compute_true_activity_fractions (exact,
    non-duplicated, straight from the original whole-body source image),
    cached on `log` itself so repeated per-bed calls within the same
    reconstruction don't re-read/re-sum that image every time. Falls back
    to the older .smi-derived approximation (see _segment_raw_count --
    double-counts overlap territory, kept only for the case where the
    original source_nifti is no longer available, e.g. moved/deleted after
    the build) if that fails.

    Returns None if the log or the needed fields aren't available (older
    logs) — the caller decides the fallback.
    """
    if log is None:
        return None

    cache = log.get("_true_activity_fractions")
    if cache is None:
        cache = _compute_true_activity_fractions(log)
        log["_true_activity_fractions"] = cache
    if cache:
        frac = cache.get(str(bed).lower())
        if frac is not None:
            return frac

    # fallback: old .smi-derived approximation
    seg = _find_segment(log, bed)
    if seg is None:
        return None
    this_raw = _segment_raw_count(seg)
    if this_raw is None:
        return None
    total = 0.0
    for s in log.get("segments", []):
        r = _segment_raw_count(s)
        if r is not None:
            total += r
    if total <= 0:
        return None
    return this_raw / total


# --------------------------------------------------------------- io helpers
def resolve_peak_files(bed, peak, log=None, workdir=None):
    """Return (base_path_without_suffix, {"peak":, "lower":, "upper":}) for
    one bed+peak, aware of whether the build combined every peak into ONE
    SIMIND run (combine_windows -- see simind_lu177_v9.py, used for any
    isotope with 2+ peaks e.g. lu177/ac225) or ran each peak as its own
    separate simulation (legacy, and still what happens for single-peak
    isotopes like tc99m/i131).

    Combined builds: every peak shares ONE output file set named
    "{bed}_tot_wN.h00" (no peak suffix in the filename at all) -- which
    window number N holds THIS peak's photopeak/lower/upper comes from the
    segment's window_offsets in the build log (1-based, matching SIMIND's
    own _tot_wN numbering).

    Legacy builds (or a combined-aware lookup finding nothing for this bed/
    peak -- e.g. no log passed, an older log with no window_offsets field,
    or a peak not present in it): each peak has its own output set named
    "{bed}_{peak}_tot_wN.h00", windows always 1 (peak) / 2 (lower) / 3
    (upper) -- this is the original, unconditional behavior, so builds made
    before combine_windows existed keep working with no changes needed.
    """
    workdir = workdir if workdir is not None else WORKDIR
    seg = _find_segment(log, bed)
    offsets = (seg.get("window_offsets") or {}) if seg else {}
    if peak in offsets:
        win = offsets[peak]
        base = os.path.join(workdir, bed)
    else:
        win = {"peak": 1, "lower": 2, "upper": 3}
        base = os.path.join(workdir, f"{bed}_{peak}")
    return base, win


def bed_paths(bed, peak=None, log=None):
    """peak: override PEAK for this call -- used by reconstruct_bed_multi_peak
    to look up several peaks' files for the SAME bed.
    log: the build log (see load_build_log) -- passed through to
    resolve_peak_files so a combine_windows build (one shared SIMIND run for
    all peaks) resolves to the right shared file + window number instead of
    the legacy per-peak-file assumption. log=None keeps the old behavior
    exactly (fully backward compatible with builds/callers that don't know
    about this)."""
    peak = peak or PEAK
    base, win = resolve_peak_files(bed, peak, log=log)
    amap_candidates = [f"{base}.hct", os.path.join(WORKDIR, f"{bed}.hct")]
    amap = next((c for c in amap_candidates if os.path.exists(c)),
                amap_candidates[0])
    p = {"photopeak": f"{base}_tot_w{win['peak']}.h00", "amap": amap}
    if USE_TEW:
        p["lower"] = f"{base}_tot_w{win['lower']}.h00"
        p["upper"] = f"{base}_tot_w{win['upper']}.h00"
    return p


def recon_output_dir(workdir, algorithm, n_iters, n_subsets, time_per_proj_s,
                     activity_mbq):
    """Every reconstruction's outputs go into a subfolder of `workdir` named
    after the algorithm and parameters that produced them, instead of
    writing straight into the simulation folder — so re-reconstructing the
    same build with different settings (algorithm, iterations, subsets,
    acquisition time, activity) keeps each result side by side instead of
    overwriting the last one.
    Name format: {algorithm}--{n_iters}iter-{n_subsets}sub--{time}sec--{activity}MBq
    (time/activity rounded to a readable value, not full precision)."""
    name = (f"{algorithm.lower()}--{n_iters}iter-{n_subsets}sub--"
            f"{time_per_proj_s:g}sec--{round(activity_mbq)}MBq")
    out = os.path.join(workdir, name)
    os.makedirs(out, exist_ok=True)
    return out


def to_sitk(vol_xyz, object_meta, frame, flip_axial=True):
    """Wrap a reconstructed (x,y,z) numpy volume as a SimpleITK image with the
    correct world frame. `frame` = (origin_mm_xyz, spacing_mm_xyz, direction9)
    from the build log; if None, fall back to the SPECT voxel size from
    object_meta and origin 0. SimpleITK expects the array in (z,y,x) order.

    flip_axial: SIMIND flips voxel phantoms axially on read, so reconstructions
    come out flipped along z relative to the input PET/CT. The correction,
    verified by the user, is a SimpleITK axial flip that preserves the world
    geometry:
        out = sitk.Flip(img, [False, False, True])
        out.SetOrigin(img.GetOrigin()); out.SetDirection(...); out.SetSpacing(...)
    i.e. flip in index space, keep the original origin/spacing/direction so the
    volume overlays on the original image. If overlay is still upside-down,
    set flip_axial=False."""
    # PyTomography object volume is (x,y,z); SimpleITK wants (z,y,x).
    arr = np.ascontiguousarray(np.transpose(vol_xyz, (2, 1, 0)).astype(np.float32))
    img = sitk.GetImageFromArray(arr)

    # Reconstructed voxel size = SPECT object voxel size (cm -> mm).
    dr = object_meta.dr  # (dx,dy,dz) in cm per PyTomography
    recon_spacing_mm = [float(dr[0]) * 10.0, float(dr[1]) * 10.0,
                        float(dr[2]) * 10.0]

    if frame is not None and frame[0] is not None:
        origin, prep_spacing, direction = frame
        img.SetOrigin([float(o) for o in origin])
        img.SetSpacing(recon_spacing_mm)
        img.SetDirection([float(d) for d in direction])
    else:
        img.SetSpacing(recon_spacing_mm)
        img.SetOrigin([0.0, 0.0, 0.0])

    if flip_axial:
        # user-verified correction: flip the z axis, keep the geometry.
        o, d, s = img.GetOrigin(), img.GetDirection(), img.GetSpacing()
        img = sitk.Flip(img, [False, False, True])
        img.SetOrigin(o)
        img.SetDirection(d)
        img.SetSpacing(s)
    return img


def load_amap_robust(hct_path, object_meta, convert, factor):
    """Read a SIMIND aligned attenuation map at its TRUE dimensions and resample
    it to the reconstructed object grid.

    PyTomography's get_attenuation_map trusts the header's matrix size [3], but
    SIMIND may write the aligned map with a DIFFERENT axial slice count than the
    reconstructed object (it resamples the map to the SPECT slice thickness,
    which can differ from the phantom voxel size — e.g. a 68-slice phantom can
    yield a 34-slice map). That mismatch is the 'cannot reshape' error. Here we
    read the real (nx,ny,nz) from the file size and header, then resample the
    map to the object's (matrix, matrix, matrix) grid so attenuation correction
    is geometrically aligned.
    """
    import re
    import torch as _torch
    hdr = open(hct_path, "r", errors="ignore").read()
    def key(k, default=None):
        m = re.search(rf"{k}\s*:=\s*(\S+)", hdr)
        return m.group(1) if m else default
    ict = hct_path.rsplit(".", 1)[0] + ".ict"
    if not os.path.exists(ict):
        nm = key(r"!name of data file")
        if nm:
            ict = os.path.join(os.path.dirname(hct_path), os.path.basename(nm))
    nx = int(key(r"matrix size \[1\]"))
    ny = int(key(r"matrix size \[2\]"))
    bpp = int(key(r"number of bytes per pixel", "4"))
    dtype = {2: "<i2", 4: "<f4"}[bpp]
    raw = np.fromfile(ict, dtype=dtype)
    nz = raw.size // (nx * ny)           # TRUE slice count from file size
    vol = raw.reshape(nz, ny, nx).astype(np.float32)   # (z,y,x)
    vol = np.transpose(vol, (2, 1, 0))                 # -> (x,y,z)
    if convert:
        vol = vol * factor

    # target object grid from PyTomography meta
    tgt = tuple(int(s) for s in object_meta.shape)     # (x,y,z)
    if vol.shape != tgt:
        # resample (nearest along each axis via linear interp on a grid).
        # attenuation maps are smooth enough; use scipy if available, else a
        # simple index remap.
        try:
            from scipy.ndimage import zoom
            factors = [t / s for t, s in zip(tgt, vol.shape)]
            vol = zoom(vol, factors, order=1)
        except Exception:
            # fallback: nearest-neighbour index remap per axis
            idx = [np.clip((np.arange(t) * s / t).astype(int), 0, s - 1)
                   for t, s in zip(tgt, vol.shape)]
            vol = vol[np.ix_(idx[0], idx[1], idx[2])]
    print(f"  amap: file {(nx, ny, nz)} -> object {tgt} "
          f"(dtype {dtype}, {'mu' if not convert else 'converted'})")
    return _torch.as_tensor(np.ascontiguousarray(vol.astype(np.float32)))


# --------------------------------------------------------------- recon core
_PRIOR_CLASSES = {"relative_difference": RelativeDifferencePrior,
                  "quadratic": QuadraticPrior}
_LIKELIHOOD_ALGORITHMS = {"OSEM": OSEM, "MLEM": MLEM}
_PRIOR_ALGORITHMS = {"OSMAPOSL": OSMAPOSL, "BSREM": BSREM,
                     "RBIEM": RBIEM, "RBIMAP": RBIMAP}
_ALGORITHMS_REQUIRING_PRIOR = {"OSMAPOSL", "BSREM", "RBIMAP"}   # RBIEM: optional
# "FBP" deliberately excluded -- see the NotImplementedError below for why
# (confirmed broken in the installed pytomography==3.3.2, not our bug).
ALL_ALGORITHMS = sorted(set(_LIKELIHOOD_ALGORITHMS) | set(_PRIOR_ALGORITHMS)
                        | {"SART"})


def _build_prior():
    """RelativeDifferencePrior/QuadraticPrior from PRIOR_TYPE/PRIOR_BETA/
    PRIOR_GAMMA, or None if PRIOR_BETA is unset (no regularization)."""
    if PRIOR_BETA is None:
        return None
    key = (PRIOR_TYPE or "relative_difference").lower()
    if key not in _PRIOR_CLASSES:
        raise ValueError(f"PRIOR_TYPE must be one of {list(_PRIOR_CLASSES)}, "
                         f"got {PRIOR_TYPE!r}")
    cls = _PRIOR_CLASSES[key]
    if cls is RelativeDifferencePrior:
        return cls(beta=PRIOR_BETA, gamma=PRIOR_GAMMA)
    return cls(beta=PRIOR_BETA)   # QuadraticPrior has no gamma


def _run_with_system_matrix(system_matrix, photopeak_realization, additive_term):
    """Run ALGORITHM (see the CONFIG block's docstring for the full list and
    what each needs) against an already-built system_matrix. Shared by the
    single-peak path (_run_recon_algorithm builds a plain SPECTSystemMatrix
    and calls this) and the multi-peak/dual-window path
    (reconstruct_bed_multi_peak builds an ExtendedSystemMatrix and calls
    this directly) — same algorithm dispatch either way.
    """
    algo = (ALGORITHM or "OSEM").upper()

    if algo in _LIKELIHOOD_ALGORITHMS:
        likelihood = PoissonLogLikelihood(system_matrix=system_matrix,
                                          projections=photopeak_realization,
                                          additive_term=additive_term)
        recon_algorithm = _LIKELIHOOD_ALGORITHMS[algo](likelihood)
        if algo == "MLEM":       # MLEM has no subsets
            return recon_algorithm(n_iters=N_ITERS).cpu().numpy()
        return recon_algorithm(n_iters=N_ITERS, n_subsets=N_SUBSETS).cpu().numpy()

    if algo in _PRIOR_ALGORITHMS:
        likelihood = PoissonLogLikelihood(system_matrix=system_matrix,
                                          projections=photopeak_realization,
                                          additive_term=additive_term)
        prior = _build_prior()
        if prior is None and algo in _ALGORITHMS_REQUIRING_PRIOR:
            raise ValueError(
                f"{algo} needs a prior — set PRIOR_BETA (reconstruct()'s "
                f"prior_beta=) before using it. RBIEM is the one algorithm "
                f"in this family that works without one.")
        kwargs = {"prior": prior}
        if algo == "BSREM" and RELAXATION_SEQUENCE is not None:
            kwargs["relaxation_sequence"] = RELAXATION_SEQUENCE
        recon_algorithm = _PRIOR_ALGORITHMS[algo](likelihood=likelihood, **kwargs)
        return recon_algorithm(n_iters=N_ITERS, n_subsets=N_SUBSETS).cpu().numpy()

    if algo == "SART":
        kwargs = {}
        if RELAXATION_SEQUENCE is not None:
            kwargs["relaxation_sequence"] = RELAXATION_SEQUENCE
        recon_algorithm = SART(system_matrix=system_matrix,
                               projections=photopeak_realization,
                               additive_term=additive_term, **kwargs)
        return recon_algorithm(n_iters=N_ITERS, n_subsets=N_SUBSETS).cpu().numpy()

    if algo == "FBP":
        # Confirmed broken in the installed pytomography==3.3.2:
        # FilteredBackProjection.__init__ never stores `projections` as
        # self.proj despite __call__ reading self.proj (AttributeError), and
        # even past that, __call__ does len(proj_meta.shape[0]) on an int
        # (TypeError). Both are library bugs, not something fixable from
        # here without monkey-patching internals we can't verify are then
        # mathematically correct -- so FBP is disabled until pytomography
        # ships a fix. Everything else in ALL_ALGORITHMS was verified to
        # actually run correctly against this install.
        raise NotImplementedError(
            "FBP is disabled: pytomography==3.3.2's FilteredBackProjection "
            "has confirmed bugs (missing self.proj assignment, then a "
            "len() TypeError) that make it not run at all in this install. "
            "Use OSEM/MLEM/OSMAPOSL/BSREM/RBIEM/RBIMAP/SART instead, or "
            "check for a newer pytomography version.")

    raise ValueError(f"ALGORITHM must be one of {ALL_ALGORITHMS}, got {ALGORITHM!r}")


def _run_recon_algorithm(object_meta, proj_meta, photopeak_realization,
                         obj_transforms, additive_term):
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms=obj_transforms,
        proj2proj_transforms=[],
        object_meta=object_meta,
        proj_meta=proj_meta)
    return _run_with_system_matrix(system_matrix, photopeak_realization,
                                   additive_term)


def reconstruct_bed(bed, activity_scale=1.0, log=None):
    """Reconstruct FOUR variants for one bed:
        NC     : no attenuation, no scatter
        AC     : attenuation only
        SC     : scatter only  (needs TEW windows)
        ACSC   : attenuation + scatter
    Returns (dict of variant->volume, object_meta, projection_realization).
    Variants needing scatter are skipped (with a note) if USE_TEW is False.

    activity_scale: per-bed correction (see segment_activity_scale) applied to
    the photopeak/scatter windows before the Poisson draw, so beds simulated
    with different per-segment source normalization come back on a common,
    physically comparable brightness scale.
    log: build log, forwarded to bed_paths (see its docstring) so a
    combine_windows build resolves to the right shared output file/window.
    """
    paths = bed_paths(bed, log=log)
    if not os.path.exists(paths["photopeak"]):
        raise FileNotFoundError(
            f"{paths['photopeak']} not found — re-run SIMIND with /tr:14 "
            f"(Interfile). You currently have .mhd headers.")
    if not os.path.exists(paths["amap"]):
        raise FileNotFoundError(
            f"{paths['amap']} not found — re-run SIMIND with /tr:15 for the "
            f"aligned attenuation map.")

    print_diagnostics(paths["photopeak"])

    dist_unit = radius_distance_unit(paths["photopeak"])
    object_meta, proj_meta = simind.get_metadata(paths["photopeak"],
                                                  distance=dist_unit)
    print(f"object_meta.shape: {object_meta.shape}  dr: {object_meta.dr}  "
          f"radius: {proj_meta.radii[0]:.1f} cm "
          f"(header units auto-detected as {dist_unit})")
    photopeak = simind.get_projections(paths["photopeak"])
    if ORIENT_PROJECTIONS_FOR_RECON:
        photopeak = orient_projection(photopeak)
    photopeak_realization = torch.poisson(
        photopeak * ACTIVITY_MBQ * TIME_PER_PROJ_S * activity_scale)

    # scatter estimate (TEW) — only if lower/upper windows exist
    additive_term = None
    if USE_TEW:
        lower = simind.get_projections(paths["lower"])
        upper = simind.get_projections(paths["upper"])
        lower_r = torch.poisson(lower * ACTIVITY_MBQ * TIME_PER_PROJ_S * activity_scale)
        upper_r = torch.poisson(upper * ACTIVITY_MBQ * TIME_PER_PROJ_S * activity_scale)
        ww_peak = simind.get_energy_window_width(paths["photopeak"])
        ww_lower = simind.get_energy_window_width(paths["lower"])
        ww_upper = simind.get_energy_window_width(paths["upper"])
        additive_term = simind.compute_EW_scatter(
            lower_r, upper_r, ww_lower, ww_upper, ww_peak)

    # attenuation transform (robust reader handles slice-count mismatch)
    amap = load_amap_robust(paths["amap"], object_meta,
                            CONVERT_DENSITY_TO_MU, DENSITY_TO_MU_FACTOR)
    att_transform = SPECTAttenuationTransform(amap)

    # PSF (collimator response) — used in ALL variants; it's not a "correction"
    # toggle, it's the physics of the detector. Keep it on everywhere.
    psf_meta = simind.get_psfmeta_from_header(paths["photopeak"])
    psf_transform = SPECTPSFTransform(psf_meta)

    variants = {}
    # NC: PSF only, no attenuation, no scatter
    print("  reconstructing NC   (no att, no scatter) ...")
    variants["NC"] = _run_recon_algorithm(object_meta, proj_meta, photopeak_realization,
                               [psf_transform], None)
    # AC: attenuation + PSF, no scatter
    print("  reconstructing AC   (attenuation) ...")
    variants["AC"] = _run_recon_algorithm(object_meta, proj_meta, photopeak_realization,
                               [att_transform, psf_transform], None)
    if additive_term is not None:
        # SC: scatter + PSF, no attenuation
        print("  reconstructing SC   (scatter) ...")
        variants["SC"] = _run_recon_algorithm(object_meta, proj_meta,
                                   photopeak_realization,
                                   [psf_transform], additive_term)
        # ACSC: attenuation + scatter + PSF
        print("  reconstructing ACSC (attenuation + scatter) ...")
        variants["ACSC"] = _run_recon_algorithm(object_meta, proj_meta,
                                     photopeak_realization,
                                     [att_transform, psf_transform],
                                     additive_term)
    else:
        print("  SC / ACSC skipped: USE_TEW is False (no scatter windows). "
              "Add lower/upper windows to lu177.win and set USE_TEW=True.")

    return variants, object_meta, photopeak_realization


def reconstruct_bed_multi_peak(bed, peaks, activity_scale=1.0,
                               calibration_factors=None, mu_waters=None,
                               log=None):
    """Combine SEVERAL energy-window peaks into ONE joint reconstruction for
    this bed — PyTomography's "dual energy window" approach generalized to
    any number of peaks:
    https://pytomography.readthedocs.io/en/stable/notebooks/t_dualpeak.html

    Each peak gets its OWN PSF and attenuation transform (both are energy-
    dependent), each scaled by that peak's calibration_factor (counts/
    second/MBq — accounts for different detection efficiency between
    energies), then all peaks' system matrices are combined via
    ExtendedSystemMatrix and reconstructed TOGETHER into ONE set of
    NC/AC/SC/ACSC images — not one set per peak. This differs from just
    reconstructing each peak separately and averaging/adding the images
    afterward: PyTomography's forward model accounts for each peak's own
    geometry/attenuation/PSF simultaneously during every iteration.

    peaks: list of resolved peak tag strings (e.g. ["113kev", "208kev"]) —
        already resolved (via resolve_peak) if using the {"center_kev":...}
        spec form; this function only takes plain tags.
    calibration_factors: list matching `peaks`, or None (all 1.0 — only
        physically correct if you don't need cross-peak relative weighting
        corrected; pass real calibration factors for quantitative combined
        reconstruction).
    mu_waters: list matching `peaks`, each an explicit cm^2/g water
        attenuation coefficient, or None per-peak to use MU_WATER_TABLE's
        built-in NIST value (only defined for 113/208 keV Lu-177 peaks —
        required explicitly for any other peak).
    log: build log, forwarded to bed_paths per peak (see its docstring) so a
        combine_windows build (one shared SIMIND run for all peaks) resolves
        each peak to its own window within the shared output file, instead
        of assuming the legacy per-peak-file convention.

    Returns (variants, object_meta, photopeak_realizations) — like
    reconstruct_bed, except photopeak_realizations is a LIST (one torch
    tensor per peak, in `peaks` order) instead of a single tensor, since
    each peak still has its own raw projection data worth inspecting.
    """
    n = len(peaks)
    if calibration_factors is None:
        calibration_factors = [1.0] * n
    if mu_waters is None:
        mu_waters = [None] * n
    if len(calibration_factors) != n or len(mu_waters) != n:
        raise ValueError("calibration_factors/mu_waters must be the same "
                         "length as peaks")

    object_meta_ref = proj_meta_ref = None
    photopeak_realizations = []
    additive_terms = []
    sysmats = {"NC": [], "AC": [], "SC": [], "ACSC": []}
    have_tew = False

    for peak, calib, mu_w_override in zip(peaks, calibration_factors, mu_waters):
        paths = bed_paths(bed, peak=peak, log=log)
        if not os.path.exists(paths["photopeak"]):
            raise FileNotFoundError(
                f"{paths['photopeak']} not found — multi-peak reconstruction "
                f"needs EVERY peak in `peaks` actually simulated for {bed}.")
        if not os.path.exists(paths["amap"]):
            raise FileNotFoundError(f"{paths['amap']} not found for {bed} @ {peak}.")

        print_diagnostics(paths["photopeak"])
        dist_unit = radius_distance_unit(paths["photopeak"])
        object_meta, proj_meta = simind.get_metadata(paths["photopeak"],
                                                      distance=dist_unit)
        if object_meta_ref is None:
            object_meta_ref, proj_meta_ref = object_meta, proj_meta
        print(f"  [{peak}] object_meta.shape: {object_meta.shape}  "
              f"radius: {proj_meta.radii[0]:.1f} cm  calibration={calib:g}")

        photopeak = simind.get_projections(paths["photopeak"])
        if ORIENT_PROJECTIONS_FOR_RECON:
            photopeak = orient_projection(photopeak)
        photopeak_realization = torch.poisson(
            photopeak * ACTIVITY_MBQ * TIME_PER_PROJ_S * activity_scale)
        photopeak_realizations.append(photopeak_realization)

        additive_term = None
        if USE_TEW and "lower" in paths and os.path.exists(paths["lower"]):
            lower = simind.get_projections(paths["lower"])
            upper = simind.get_projections(paths["upper"])
            lower_r = torch.poisson(lower * ACTIVITY_MBQ * TIME_PER_PROJ_S * activity_scale)
            upper_r = torch.poisson(upper * ACTIVITY_MBQ * TIME_PER_PROJ_S * activity_scale)
            ww_peak = simind.get_energy_window_width(paths["photopeak"])
            ww_lower = simind.get_energy_window_width(paths["lower"])
            ww_upper = simind.get_energy_window_width(paths["upper"])
            additive_term = simind.compute_EW_scatter(
                lower_r, upper_r, ww_lower, ww_upper, ww_peak)
            have_tew = True
        additive_terms.append(additive_term)

        mu_w = mu_w_override if mu_w_override is not None else MU_WATER_TABLE.get(peak)
        if CONVERT_DENSITY_TO_MU and mu_w is None:
            raise ValueError(
                f"No built-in NIST mu_water for peak {peak!r}; pass "
                f"mu_waters=[...] with an explicit cm^2/g value for it.")
        density_to_mu_factor = (mu_w / 1000.0) if CONVERT_DENSITY_TO_MU else 1.0
        amap = load_amap_robust(paths["amap"], object_meta,
                                CONVERT_DENSITY_TO_MU, density_to_mu_factor)
        att_transform = SPECTAttenuationTransform(amap)
        psf_meta = simind.get_psfmeta_from_header(paths["photopeak"])
        psf_transform = SPECTPSFTransform(psf_meta)

        def sysmat(transforms):
            return calib * SPECTSystemMatrix(
                obj2obj_transforms=transforms, proj2proj_transforms=[],
                object_meta=object_meta, proj_meta=proj_meta)

        sysmats["NC"].append(sysmat([psf_transform]))
        sysmats["AC"].append(sysmat([att_transform, psf_transform]))
        if additive_term is not None:
            sysmats["SC"].append(sysmat([psf_transform]))
            sysmats["ACSC"].append(sysmat([att_transform, psf_transform]))

    if have_tew and any(a is None for a in additive_terms):
        raise ValueError(
            "multi-peak reconstruction needs TEW scatter windows for EVERY "
            "peak in `peaks`, or NONE of them — got a mix.")

    stacked_photopeak = torch.stack(photopeak_realizations, dim=0)
    stacked_additive = torch.stack(additive_terms, dim=0) if have_tew else None

    variants = {}
    print(f"  reconstructing NC   (no att, no scatter) [{n}-peak joint] ...")
    variants["NC"] = _run_with_system_matrix(
        ExtendedSystemMatrix(system_matrices=sysmats["NC"]), stacked_photopeak, None)
    print(f"  reconstructing AC   (attenuation) [{n}-peak joint] ...")
    variants["AC"] = _run_with_system_matrix(
        ExtendedSystemMatrix(system_matrices=sysmats["AC"]), stacked_photopeak, None)
    if have_tew:
        print(f"  reconstructing SC   (scatter) [{n}-peak joint] ...")
        variants["SC"] = _run_with_system_matrix(
            ExtendedSystemMatrix(system_matrices=sysmats["SC"]),
            stacked_photopeak, stacked_additive)
        print(f"  reconstructing ACSC (attenuation + scatter) [{n}-peak joint] ...")
        variants["ACSC"] = _run_with_system_matrix(
            ExtendedSystemMatrix(system_matrices=sysmats["ACSC"]),
            stacked_photopeak, stacked_additive)
    else:
        print("  SC / ACSC skipped: no TEW scatter windows for these peaks.")

    return variants, object_meta_ref, photopeak_realizations


def segment_crop(log, bed):
    """Return (data_z0, data_z1) voxel bounds of the real (non-pad) data within
    a bed's reconstructed volume, or None if the log has no crop metadata
    (older logs predating this field, or log missing entirely)."""
    seg = _find_segment(log, bed)
    if seg is None:
        return None
    r = seg.get("data_z_range")
    if r and len(r) == 2:
        return int(r[0]), int(r[1])
    return None


def crop_bed_to_data(vol_xyz, crop):
    """Crop a (x,y,z) volume's z axis to the real-data window, discarding the
    air-padding rows added during simulation prep."""
    if crop is None:
        return vol_xyz
    z0, z1 = crop
    z0 = max(0, min(z0, vol_xyz.shape[2]))
    z1 = max(z0, min(z1, vol_xyz.shape[2]))
    return np.ascontiguousarray(vol_xyz[:, :, z0:z1])


def frame_shift_z(frame, n_vox, dz_mm):
    """Shift a (origin_mm_xyz, spacing, direction) frame's z-origin forward by
    n_vox voxels, to compensate for cropping n_vox low-pad slices off a bed."""
    if frame is None or frame[0] is None:
        return frame
    origin = list(frame[0])
    origin[2] = float(origin[2]) + n_vox * dz_mm
    return (origin, frame[1], frame[2])


def stitch_axial(volumes, overlap_voxels):
    """Feather-merge (x,y,z) volumes along z (axis 2). Legacy path; assumes each
    volume is exactly the data with no air padding. Kept for the no-log case."""
    merged = volumes[0]
    for vol in volumes[1:]:
        o = overlap_voxels
        if o > 0:
            a = merged[..., -o:]
            b = vol[..., :o]
            w = np.linspace(1, 0, o)[None, None, :]
            blend = a * w + b * (1 - w)
            merged = np.concatenate(
                [merged[..., :-o], blend, vol[..., o:]], axis=2)
        else:
            merged = np.concatenate([merged, vol], axis=2)
    return merged


def stitch_by_origin(volumes, z_origins_mm, dz_mm):
    """Place each (x,y,z) bed onto a shared canvas at its true world z-origin,
    resolving overlaps with a HARD CUT at the midpoint of each overlap
    instead of averaging -- matching PyTomography's own official
    dicom.stitch_multibed() (verified directly against the installed
    pytomography package's real source, 2026-08-12), NOT a blend/feather.
    Each bed simply owns its slices up to the midpoint of any overlap with
    its neighbour, the neighbour owns the rest -- no voxel is ever written
    by more than one bed.

    This deliberately replaces an earlier averaging-based version. Averaging
    two INDEPENDENT noisy Poisson realizations of the same physical region
    measurably lowers the noise specifically within the overlap band
    relative to the rest of each bed (fewer effective counts everywhere
    else) -- visible as a seam/stitch right at the overlap boundary, which
    is exactly the artifact this replaces. PyTomography's own source
    comment on why they do it this way: "Only offer midslice stitch now
    because [averaging] messes with uncertainty estimation" -- same
    reasoning applies here.

    volumes      : list of (x,y,z) arrays, one per bed (z-height may differ
                   per bed after crop_bed_to_data -- not required uniform).
    z_origins_mm : world z origin (mm) of each bed's FIRST slice, same
                   order as `volumes`.
    dz_mm        : recon axial voxel size (mm).
    Returns (whole_xyz, canvas_origin_z_mm).
    """
    # sort beds by z-origin ascending -- PyTomography's own stitch_multibed
    # does this defensively too, rather than trusting caller order.
    order = sorted(range(len(volumes)), key=lambda i: z_origins_mm[i])
    volumes = [volumes[i] for i in order]
    z_origins_mm = [z_origins_mm[i] for i in order]

    # convert origins to integer voxel offsets relative to the min origin
    z0 = z_origins_mm[0]
    offs = [int(round((z - z0) / dz_mm)) for z in z_origins_mm]
    # beds may differ in height after per-bed pad cropping, so size the canvas
    # and each bed's write window off that bed's own shape, not a shared nz.
    total_z = max(o + v.shape[2] for v, o in zip(volumes, offs))
    xy = volumes[0].shape[:2]

    # binary (0/1) weight per bed -- starts as "owns everything", then each
    # overlap with the NEXT bed (beds are sorted, so only adjacent pairs can
    # overlap) is resolved by zeroing this bed from the overlap's midpoint
    # onward, and zeroing the next bed from its start up to that same
    # midpoint. No voxel ends up covered by more than one bed's weight=1.
    weights = [np.ones(v.shape, dtype=np.float32) for v in volumes]
    for i in range(len(volumes) - 1):
        nz_i = volumes[i].shape[2]
        # bed i+1's start, expressed in bed i's OWN local index (offs are
        # GLOBAL canvas offsets, so this must subtract offs[i] -- using
        # offs[i+1] directly here was a bug: correct only when offs[i]
        # happens to be 0, i.e. only for the very first pair in a chain).
        overlap_lo = offs[i + 1] - offs[i]
        overlap_hi = nz_i             # bed i's last slice, local index
        if overlap_lo >= overlap_hi:
            continue                  # these two beds don't actually overlap
        half = overlap_lo + (overlap_hi - overlap_lo) // 2   # bed i's local index
        weights[i][:, :, half:overlap_hi] = 0.0
        # convert `half` to a global slice, then into bed i+1's own local index
        local_cut = (offs[i] + half) - offs[i + 1]
        weights[i + 1][:, :, 0:max(0, local_cut)] = 0.0

    acc = np.zeros((*xy, total_z), dtype=np.float64)
    for vol, w, off in zip(volumes, weights, offs):
        nz = vol.shape[2]
        acc[..., off:off + nz] += vol * w
    return acc.astype(np.float32), z0


def compute_overlap_calibration(volumes, z_origins_mm, dz_mm, min_signal_frac=0.05):
    """Per-bed scalar calibration factor that makes ADJACENT beds' own
    independent reconstructions AGREE in their shared overlap zone, chained
    sequentially down the bed sequence (the first bed, by z-origin, is
    anchored at factor 1.0; each subsequent bed is scaled to match its
    ALREADY-calibrated predecessor, so corrections propagate consistently
    down a chain of 3+ beds rather than each being independently corrected
    against a single reference).

    WHY THIS EXISTS: activity_mbq_mode's per-bed scale (whether
    "total_by_counts" or "per_bed") is a single scalar applied UNIFORMLY
    across an entire bed, correcting for a GLOBAL bed-wide property (total
    simulated counts, or peak-voxel ratio). The visible multi-bed stitching
    seam turned out (confirmed empirically, 2026-08-12, psma_000000) to be
    a LOCAL discrepancy specific to the overlap region itself -- comparing
    two beds' independent reconstructions of the identical physical overlap
    showed a real, consistent 25-33% brightness mismatch that NEITHER
    global scalar could fix (verified: correcting a real double-counting
    bug in total_by_counts moved the measured mismatch by about 1%, nowhere
    near the ~25-33% observed; per_bed mode measured WORSE, ~2.5-3.3x).
    Since no single whole-bed metric reliably predicts the overlap-zone
    relationship, this instead calibrates directly against the one thing
    that SHOULD agree: what the two beds each independently reconstructed
    for the exact same physical region.

    Per adjacent pair, the correction is the MEDIAN of per-voxel ratios
    (bed i's calibrated value / bed i+1's raw value) at matching world
    positions within their overlap, restricted to voxels where BOTH beds
    show real signal there (> min_signal_frac of that bed's own 99th
    percentile within the overlap) -- keeps near-zero/noise-dominated
    background voxels from dominating the median with unstable ratios.

    volumes, z_origins_mm, dz_mm: same as stitch_by_origin (volumes may
    differ in z-height per bed, e.g. after crop_bed_to_data).
    Returns {original_index: factor} keyed by `volumes`' ORIGINAL (pre-sort)
    order, so the caller can directly do `vol * factors[i]` without needing
    to know about this function's internal z-origin sorting. A bed pair
    with no actual overlap, or no usable signal in it, leaves the LATER
    bed's factor at whatever its predecessor chain already established
    (i.e. effectively uncalibrated against that specific neighbour).
    """
    order = sorted(range(len(volumes)), key=lambda i: z_origins_mm[i])
    vols_sorted = [volumes[i] for i in order]
    origins_sorted = [z_origins_mm[i] for i in order]
    offs = [int(round((z - origins_sorted[0]) / dz_mm)) for z in origins_sorted]

    factors_sorted = [1.0] * len(vols_sorted)
    for i in range(len(vols_sorted) - 1):
        nz_i = vols_sorted[i].shape[2]
        overlap_lo = offs[i + 1] - offs[i]      # bed i's own local index
        overlap_hi = nz_i
        if overlap_lo >= overlap_hi:
            continue                            # these two beds don't actually overlap
        n_overlap = overlap_hi - overlap_lo
        seg_i = vols_sorted[i][:, :, overlap_lo:overlap_hi].astype(np.float64)
        seg_i1 = vols_sorted[i + 1][:, :, 0:min(n_overlap, vols_sorted[i + 1].shape[2])].astype(np.float64)
        n = min(seg_i.shape[2], seg_i1.shape[2])
        if n <= 0:
            continue
        seg_i, seg_i1 = seg_i[:, :, :n], seg_i1[:, :, :n]
        thresh_i = np.percentile(seg_i, 99) * min_signal_frac if seg_i.size else 0.0
        thresh_i1 = np.percentile(seg_i1, 99) * min_signal_frac if seg_i1.size else 0.0
        mask = (seg_i > thresh_i) & (seg_i1 > thresh_i1)
        if not np.any(mask):
            continue                            # no usable signal here -- leave uncalibrated
        # bed i is already at its own chained scale (factors_sorted[i]); this
        # finds how much bed i+1 needs multiplying to MATCH bed i's
        # (already-calibrated) values at the same physical locations.
        ratios = (seg_i[mask] * factors_sorted[i]) / seg_i1[mask]
        factors_sorted[i + 1] = float(np.median(ratios))

    return {order[k]: factors_sorted[k] for k in range(len(vols_sorted))}


def run_reconstruction(
    workdir=None, peak=None, *, beds=None, build_log=None,
    activity_mbq=None, activity_mbq_mode=None, time_per_proj_s=None,
    n_iters=None, n_subsets=None,
    use_tew=None, orient_projections_for_recon=None, flip_recon_axial=None,
    convert_density_to_mu=None, mu_water=None,
    algorithm=None, prior_type=None, prior_beta=None, prior_gamma=None,
    relaxation_sequence=None, calibration_factors=None, mu_waters=None,
):
    """Run the full multi-bed reconstruction pipeline and return a summary
    dict of the files written, instead of only being runnable as a script.

    Every argument is optional and, when omitted (None), falls back to this
    module's CONFIG-block globals (WORKDIR, PEAK, BEDS, ACTIVITY_MBQ, ...) —
    so `python reconstruct_lu177_multibed.py` keeps working exactly as
    before (the `__main__` block below just calls this with no overrides).
    Passing a value here overrides that global FOR THIS CALL ONLY.

    workdir: folder holding the SIMIND output + build_log_*.json for one
             scanner/view-count build (what CONFIG calls WORKDIR).
    peak:    a single peak -- "113kev"/"208kev" tag string, or a
             {"center_kev":, "lower_pct":, "upper_pct":} spec (see
             resolve_peak) -- OR a LIST of either, for a JOINT multi-peak
             ("dual energy window") reconstruction combining all of them
             into one set of NC/AC/SC/ACSC images via ExtendedSystemMatrix
             (see reconstruct_bed_multi_peak / PyTomography's dual-peak
             tutorial). A list of length 1 behaves like a single peak.
    beds:    explicit bed-tag list, or None to auto-detect from the build log.
    activity_mbq_mode: "per_bed" (default) applies the SAME activity_mbq to
        every bed, only correcting relative brightness between them. Set to
        "total_by_counts" to instead treat activity_mbq as the TOTAL across
        the WHOLE multi-bed study and split it across beds by each bed's
        share of total simulated source counts (see
        segment_count_fraction's docstring) -- e.g. activity_mbq=10000 with
        a bed holding 20% of total counts uses 2000 for that bed alone.
    algorithm, prior_type, prior_beta, prior_gamma, relaxation_sequence: pick
        and configure the reconstruction algorithm -- see the CONFIG block's
        ALGORITHM/PRIOR_* docstring comment above for the full list
        (OSEM/MLEM/OSMAPOSL/BSREM/RBIEM/RBIMAP/SART/FBP) and what each needs.
    calibration_factors, mu_waters: only used when `peak` is a list (multi-
        peak) -- per-peak lists matching its length, forwarded to
        reconstruct_bed_multi_peak (see its docstring). None -> calibration
        1.0 for every peak, mu_water from the built-in NIST table.

    Returns: {
        "workdir", "peak", "beds": [...],
        "per_bed": {bed: {variant: path, ...}, ...},
        "proj_per_bed": {bed: path, ...} for a single peak, or
                        {bed: {peak: path, ...}, ...} for multi-peak,
        "wholebody": {variant: path, ...}   # only if len(beds) > 1
        "proj_stitched": path or None,
    }
    """
    # NOTE: BEDS is deliberately NOT in this global list. When auto-detected
    # from the build log it must stay a per-call local (_beds below) — if it
    # were written back to the module global, a later call with beds=None
    # (e.g. looping over several patients) would silently reuse the FIRST
    # patient's bed list instead of re-detecting for the new workdir.
    global WORKDIR, PEAK, BUILD_LOG, ACTIVITY_MBQ, ACTIVITY_MBQ_MODE, \
        TIME_PER_PROJ_S, N_ITERS, N_SUBSETS, USE_TEW, \
        ORIENT_PROJECTIONS_FOR_RECON, FLIP_RECON_AXIAL, \
        CONVERT_DENSITY_TO_MU, MU_WATER, DENSITY_TO_MU_FACTOR, \
        ALGORITHM, PRIOR_TYPE, PRIOR_BETA, PRIOR_GAMMA, RELAXATION_SEQUENCE

    if workdir is not None:
        WORKDIR = workdir
    if peak is not None:
        PEAK = peak
    _beds = beds   # local; see the note above the `global` statement
    if build_log is not None:
        BUILD_LOG = build_log
    if algorithm is not None:
        ALGORITHM = algorithm
    if prior_type is not None:
        PRIOR_TYPE = prior_type
    if prior_beta is not None:
        PRIOR_BETA = prior_beta
    if prior_gamma is not None:
        PRIOR_GAMMA = prior_gamma
    if relaxation_sequence is not None:
        RELAXATION_SEQUENCE = relaxation_sequence

    # Load the log and resolve PEAK. PEAK may be a single tag/spec, OR a
    # list of them for a JOINT multi-peak reconstruction (see this
    # function's docstring) -- resolve each element the same way the
    # single-peak case always has (validated against what the build log
    # actually simulated), then join into one label string ("113kev+208kev")
    # used everywhere PEAK appears in a filename. _peak_list (the individual
    # resolved tags) is what actually drives reconstruction below.
    log = load_build_log()
    if isinstance(PEAK, (list, tuple)):
        _peak_list = [resolve_peak(p, log) for p in PEAK]
        PEAK = "+".join(_peak_list)
    else:
        _peak_list = [resolve_peak(PEAK, log)]
        PEAK = _peak_list[0]
    _multi_peak = len(_peak_list) > 1

    if activity_mbq is not None:
        ACTIVITY_MBQ = activity_mbq
    if activity_mbq_mode is not None:
        ACTIVITY_MBQ_MODE = activity_mbq_mode
    if time_per_proj_s is not None:
        TIME_PER_PROJ_S = time_per_proj_s
    if n_iters is not None:
        N_ITERS = n_iters
    if n_subsets is not None:
        N_SUBSETS = n_subsets
    if use_tew is not None:
        USE_TEW = use_tew
    if orient_projections_for_recon is not None:
        ORIENT_PROJECTIONS_FOR_RECON = orient_projections_for_recon
    if flip_recon_axial is not None:
        FLIP_RECON_AXIAL = flip_recon_axial
    if convert_density_to_mu is not None:
        CONVERT_DENSITY_TO_MU = convert_density_to_mu
    # MU_WATER depends on PEAK, so recompute whenever either could have
    # changed (mu_water lets you override the NIST value directly). Only
    # 113/208 keV (the two Lu-177 peaks) have a built-in NIST value; any
    # other peak (e.g. from a custom energy_windows build) needs mu_water=
    # passed explicitly, or attenuation correction would silently use the
    # wrong water attenuation coefficient. Multi-peak reconstructions look
    # this up PER PEAK inside reconstruct_bed_multi_peak instead (via
    # mu_waters=), so the single global MU_WATER doesn't apply -- skip it.
    if _multi_peak:
        MU_WATER = None
        DENSITY_TO_MU_FACTOR = None
    else:
        if mu_water is not None:
            MU_WATER = mu_water
        elif PEAK in MU_WATER_TABLE:
            MU_WATER = MU_WATER_TABLE[PEAK]
        else:
            raise ValueError(
                f"No built-in NIST mu_water for peak {PEAK!r} (only "
                f"{list(MU_WATER_TABLE)} are known) — pass mu_water=<cm^2/g> "
                f"explicitly for this peak.")
        DENSITY_TO_MU_FACTOR = MU_WATER / 1000.0

    started = datetime.datetime.now().astimezone()
    # every output from THIS call goes into a subfolder named after the
    # parameters that produced it, so re-reconstructing the same build with
    # different settings doesn't overwrite the last result.
    recon_dir = recon_output_dir(WORKDIR, ALGORITHM, N_ITERS, N_SUBSETS,
                                 TIME_PER_PROJ_S, ACTIVITY_MBQ)
    print(f"reconstruction output dir: {recon_dir}")

    result = {"workdir": WORKDIR, "recon_dir": recon_dir, "peak": PEAK,
              "per_bed": {}, "proj_per_bed": {}, "wholebody": {},
              "proj_stitched": None}
    activity_scales = {}   # bed -> scale actually used, for recon-debug.json

    if _beds is None:
        _beds = auto_beds(log)
        print(f"auto-detected BEDS from build log: {_beds}")
    result["beds"] = list(_beds)

    # per-variant accumulation across beds
    variant_vols = {}     # variant -> list of (x,y,z) volumes in bed order
    metas, frames, projs = [], [], []
    for bed in _beds:
        if ACTIVITY_MBQ_MODE == "total_by_counts":
            scale = segment_count_fraction(log, bed)
            if scale is None:
                print(f"  WARNING: {bed} has no usable count data for "
                      f"activity_mbq_mode='total_by_counts' (older log?) — "
                      f"falling back to 'per_bed' (segment_activity_scale) "
                      f"for this bed.")
                scale = segment_activity_scale(log, bed)
        else:
            scale = segment_activity_scale(log, bed)
        activity_scales[bed] = scale
        print(f"reconstructing {bed} @ {PEAK} (activity_scale={scale:.6g}, "
              f"mode={ACTIVITY_MBQ_MODE}, algorithm={ALGORITHM}) ...")
        if _multi_peak:
            variants, m, proj = reconstruct_bed_multi_peak(
                bed, _peak_list, activity_scale=scale,
                calibration_factors=calibration_factors, mu_waters=mu_waters,
                log=log)
        else:
            variants, m, proj = reconstruct_bed(bed, activity_scale=scale,
                                                 log=log)
        metas.append(m)
        projs.append(proj)
        frame = segment_frame(log, bed)
        crop = segment_crop(log, bed)
        if crop is not None:
            dz_mm_bed = float(m.dr[2]) * 10.0
            frame = frame_shift_z(frame, crop[0], dz_mm_bed)
            print(f"  cropping air pad: z {m.shape[2]} -> {crop[1] - crop[0]} "
                  f"(removed {crop[0]} low + {m.shape[2] - crop[1]} high pad rows)")
        frames.append(frame)

        # write each correction variant as its own registered NIfTI
        result["per_bed"][bed] = {}
        for name, vol in variants.items():
            vol = crop_bed_to_data(vol, crop)
            variant_vols.setdefault(name, []).append(vol)
            img = to_sitk(vol, m, frame, flip_axial=FLIP_RECON_AXIAL)
            bed_out = os.path.join(recon_dir, f"recon_{bed}_{PEAK}_{name}.nii.gz")
            sitk.WriteImage(img, bed_out)
            result["per_bed"][bed][name] = bed_out
            print(f"  saved {bed_out}")

        # projection stack for this bed (detector space, inspection). Multi-
        # peak: one file PER PEAK (proj is a list, one tensor per peak in
        # _peak_list order), each named by its own peak tag so they don't
        # collide -- proj_per_bed becomes {bed: {peak: path}} instead of
        # {bed: path} for this case (see the docstring's Returns section).
        if _multi_peak:
            result["proj_per_bed"][bed] = {}
            for peak_i, proj_i in zip(_peak_list, proj):
                proj_out = os.path.join(recon_dir, f"proj_{bed}_{peak_i}.nii.gz")
                projections_to_nifti(proj_i,
                                     bed_paths(bed, peak=peak_i, log=log)["photopeak"],
                                     proj_out)
                result["proj_per_bed"][bed][peak_i] = proj_out
                print(f"  saved {proj_out}  (angle, v, u) detector frame — "
                      f"NOT patient space")
        else:
            proj_out = os.path.join(recon_dir, f"proj_{bed}_{PEAK}.nii.gz")
            projections_to_nifti(proj, bed_paths(bed, log=log)["photopeak"], proj_out)
            result["proj_per_bed"][bed] = proj_out
            print(f"  saved {proj_out}  (angle, v, u) detector frame — "
                  f"NOT patient space")

    # ---- stitch each variant separately (multi-bed) ----------------------
    if len(_beds) > 1 and log is not None:
        dz_mm = float(metas[0].dr[2]) * 10.0
        # each bed's world z-origin (mm) from the build log
        z_origins = []
        for b, fr in zip(_beds, frames):
            if fr is not None and fr[0] is not None:
                z_origins.append(float(fr[0][2]))
            else:
                z_origins.append(float(segment_frame(log, b)[0][2]))
        print(f"  placing beds by z-origin {[round(z,1) for z in z_origins]} "
              f"mm, dz={dz_mm:.1f} mm")

        # the frame for the WHOLE volume: same x,y as a bed, z-origin = min
        base_frame = frames[0]
        whole_z0 = min(z_origins)

        # cross-calibrate beds against each other using their shared
        # overlap zones directly (see compute_overlap_calibration's
        # docstring for why this is needed on top of activity_mbq_mode's
        # own per-bed scale) -- computed ONCE from a reference variant
        # (ACSC if present, the most clinically complete correction; else
        # whichever variant exists) and applied identically to EVERY
        # variant's stitch below, so NC/AC/SC/ACSC treat each bed
        # consistently. Deliberately NOT applied to the per-bed NIfTIs
        # saved above -- those stay each bed's own independent
        # reconstruction, useful for exactly this kind of diagnosis.
        ref_name = "ACSC" if "ACSC" in variant_vols else next(iter(variant_vols))
        calibration = compute_overlap_calibration(variant_vols[ref_name], z_origins, dz_mm)
        print(f"  overlap cross-calibration factors ({ref_name}): "
              f"{[round(calibration[i], 4) for i in range(len(_beds))]}")

        for name, variant in variant_vols.items():
            calibrated = [vol * calibration[i] for i, vol in enumerate(variant)]
            whole, _ = stitch_by_origin(calibrated, z_origins, dz_mm)
            # build a frame whose z-origin is the canvas start (min origin).
            # (x,y origin/direction come from any bed; only z-origin changes.)
            frame = (list(base_frame[0][:2]) + [whole_z0],
                     base_frame[1], base_frame[2])
            whole_img = to_sitk(whole, metas[0], frame,
                                flip_axial=FLIP_RECON_AXIAL)
            out_nii = os.path.join(recon_dir,
                                   f"recon_wholebody_{PEAK}_{name}.nii.gz")
            sitk.WriteImage(whole_img, out_nii)
            result["wholebody"][name] = out_nii
            print(f"  saved {out_nii}  shape={whole.shape}")
    elif len(_beds) > 1:
        # no build log -> fall back to legacy feather stitch (may show a band
        # if beds are air-padded). Provide fov_overlap_cm-based overlap.
        dz_mm = float(metas[0].dr[2]) * 10.0
        overlap_vox = int(round((5.0 * 10.0) / dz_mm))
        print(f"  no build log; legacy feather stitch, overlap={overlap_vox}")
        for name, variant in variant_vols.items():
            whole = stitch_axial(variant[::-1], overlap_vox)
            whole_img = to_sitk(whole, metas[0], frames[0],
                                flip_axial=FLIP_RECON_AXIAL)
            out_nii = os.path.join(recon_dir,
                                   f"recon_wholebody_{PEAK}_{name}.nii.gz")
            sitk.WriteImage(whole_img, out_nii)
            result["wholebody"][name] = out_nii
            print(f"  saved {out_nii}  shape={whole.shape}")
    else:
        print(f"single bed — recon_{_beds[0]}_{PEAK}_<VARIANT>.nii.gz are the "
              f"registered outputs (NC/AC" +
              ("/SC/ACSC" if USE_TEW else "") + ")")

    # ---- stitched PROJECTION stack (detector space, inspection only) ------
    # Each bed's projections get the SAME per-frame correction as the per-bed
    # NIfTIs (orient_projection), THEN we concatenate along the stacked axis.
    # Previously the stitch used the raw uncorrected tensors, so each bed was
    # in the wrong orientation even though it stacked in the right order.
    # Still a visualization montage, NOT a co-sampled sinogram.
    # Multi-peak: each bed's `proj` is a LIST (one tensor per peak) instead
    # of a single tensor, which this montage doesn't handle -- the per-bed,
    # per-peak proj_{bed}_{peak}.nii.gz files above already cover inspection,
    # so just skip the cross-bed montage rather than guess how to combine
    # peaks and beds into one 2D concatenation.
    if _multi_peak and len(projs) > 1:
        print("  cross-bed projection montage skipped for multi-peak "
              "reconstructions — see the per-bed, per-peak proj_*.nii.gz "
              "files instead.")
    elif len(projs) > 1:
        # correct each bed, reverse to match recon bed order
        corr = [orient_projection(p).detach().cpu().numpy().astype("float32")
                for p in projs][::-1]
        if len({p.shape[0] for p in corr}) == 1 and \
           len({p.shape[2] for p in corr}) == 1:
            # trim the same overlap used for the recon stitch so beds don't
            # duplicate anatomy (or leave a gap). overlap_vox was computed above
            # from the bed z-origins; reuse it if available, else 0.
            try:
                ov = max(0, int(overlap_vox))
            except NameError:
                ov = 0
            stacked = [corr[0]]
            for p in corr[1:]:
                stacked.append(p[:, :, ov:] if ov > 0 else p)
            stitched = np.concatenate(stacked, axis=2)       # (angle,u,v_stack)
            # write with the SAME axis convention as projections_to_nifti:
            # array fed to sitk must be (angle, v, u) -> transpose (0,2,1)
            img = sitk.GetImageFromArray(
                np.ascontiguousarray(np.transpose(stitched, (0, 2, 1))))
            k = read_interfile_keys(bed_paths(_beds[0], log=log)["photopeak"])
            du = float(k.get("scaling factor (mm/pixel) [1]", "4.0"))
            dv = float(k.get("scaling factor (mm/pixel) [2]", "4.0"))
            extent = float(k.get("extent of rotation", "360"))
            nproj = int(float(k.get("number of projections", stitched.shape[0])))
            dangle = extent / nproj if nproj else 1.0
            img.SetSpacing([du, dv, dangle]); img.SetOrigin([0.0, 0.0, 0.0])
            pout = os.path.join(recon_dir, f"proj_stitched_{PEAK}.nii.gz")
            sitk.WriteImage(img, pout)
            result["proj_stitched"] = pout
            print(f"saved {pout}  shape={stitched.shape} "
                  f"(angle, u, v_stacked) — inspection montage only")
        else:
            print("projection stitch skipped: beds differ in angle count or "
                  "detector width")

    # ---- recon-debug.json: every parameter this reconstruction actually
    # used, so a result can be understood later without re-deriving it from
    # this module's CONFIG block (which may since have changed). The osem
    # parameter subfolder has no peak in its name (iterations/subsets/time/
    # activity are usually shared across peaks of the same study), so
    # reconstructing several peaks into the SAME recon_dir is expected --
    # keep one recon-debug.json per folder with a per-peak "runs" entry
    # instead of the later peak silently overwriting the earlier one's log.
    finished = datetime.datetime.now().astimezone()
    debug_path = os.path.join(recon_dir, "recon-debug.json")
    debug_log = {"schema": "reconstruct_lu177_debug_log/v1",
                 "recon_dir": recon_dir, "runs": {}}
    if os.path.exists(debug_path):
        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                debug_log = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass    # corrupt/unreadable -> start fresh rather than crash
    debug_log.setdefault("runs", {})
    debug_log["runs"][PEAK] = {
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "owner": {"name": "Yazdan Salimi", "email": "salimiyazdan@gmail.com"},
        "workdir": WORKDIR,
        "build_log_used": BUILD_LOG or os.path.join(WORKDIR, "simulation-debug.json"),
        "peak": PEAK,
        "beds": list(_beds),
        "activity_scale_per_bed": activity_scales,
        "parameters": {
            "algorithm": ALGORITHM,
            "prior_type": PRIOR_TYPE if PRIOR_BETA is not None else None,
            "prior_beta": PRIOR_BETA,
            "prior_gamma": PRIOR_GAMMA if PRIOR_BETA is not None else None,
            "n_iters": N_ITERS,
            "n_subsets": N_SUBSETS,
            "use_tew": USE_TEW,
            "activity_mbq": ACTIVITY_MBQ,
            "activity_mbq_mode": ACTIVITY_MBQ_MODE,
            "time_per_proj_s": TIME_PER_PROJ_S,
            "orient_projections_for_recon": ORIENT_PROJECTIONS_FOR_RECON,
            "flip_recon_axial": FLIP_RECON_AXIAL,
            "convert_density_to_mu": CONVERT_DENSITY_TO_MU,
            "mu_water": MU_WATER,
            "multi_peak": _multi_peak,
            "peaks": _peak_list,
            "calibration_factors": calibration_factors if _multi_peak else None,
        },
        "outputs": result,
    }
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_log, f, indent=2, ensure_ascii=False)
    print(f"saved {debug_path}")
    result["recon_debug_log"] = debug_path

    return result


# ============================================================================
#  SECTION 3 — simind_v9 CLASS  (thin facade over sections 1 & 2 above)
# ============================================================================
class simind_v9:
    """Stateful convenience wrapper. Constructor args are just per-instance
    defaults that both methods fall back to when not overridden per-call —
    nothing here is required if you'd rather pass everything explicitly each
    time, matching how the builder section's build_simind_lu177() already works.
    """

    def __init__(self, *, scanner=None, template_smc=None, smc_dir=None,
                 isotope="lu177", mpi_ranks=14, simind_bin_dir=None):
        """
        scanner:      default scanner key from SCANNER_DB (e.g.
                       "siemens_symbia_t", "ge_nm870"). Can be overridden per
                       prepare_simind_input() call; required one way or the
                       other.
        template_smc: default .smc template. Defaults to the shipped
                       C:\\simind\\v9\\simind.smc.
        smc_dir:      default SMC_DIR (isotope .isd spectra). Defaults to
                       C:\\simind\\v9\\smc_dir.
        isotope:      default isotope registry key (see ISOTOPES in
                       this file). "lu177" by default.
        mpi_ranks:    default -n for the generated run_all_mpi.bat.
        simind_bin_dir: default folder holding the `simind` executable, or
                       None (default). If set, every generated run_all.sh
                       gets `export PATH="<this>:$PATH"` prepended so it can
                       be run standalone (e.g. `bash /abs/path/run_all.sh`,
                       from any directory, on a machine where `simind` isn't
                       already on PATH — typical for a fresh shell on an
                       HPC/Linux install). Only affects run_all.sh (Linux);
                       the Windows .bat files are unaffected.
        """
        self.scanner = scanner
        self.template_smc = template_smc or os.path.join(_THIS_DIR, "simind.smc")
        self.smc_dir = smc_dir or os.path.join(_THIS_DIR, "smc_dir")
        self.isotope = isotope
        self.mpi_ranks = mpi_ranks
        self.simind_bin_dir = simind_bin_dir

    def __repr__(self):
        return (f"simind_v9(scanner={self.scanner!r}, isotope={self.isotope!r}, "
                f"mpi_ranks={self.mpi_ranks}, smc_dir={self.smc_dir!r})")

    # ------------------------------------------------------------- prepare
    def prepare_inputs(self, source_nifti, density_nifti, workdir, **kwargs):
        """Resample PET(source)/CT(density) onto a common SPECT grid, and
        convert CT HU -> density g/cm^3 if density_is_hu (default True) —
        the step that must run BEFORE prepare_simind_input(). Thin wrapper
        over the builder section's prepare_inputs(); see its docstring for
        density_is_hu / resample_spacing_mm. Returns (source_path, density_path)
        pointing at the prepared NIfTIs written into `workdir`.

        pet_interp / ct_interp kwargs (each "linear" default / "nearest" /
        "bspline") independently pick the resampling interpolation for the
        activity map and the density map — e.g. use ct_interp="nearest" to
        keep a phantom's true sharp-edged sphere values instead of getting a
        linearly-blended boundary ramp. See prepare_inputs()'s own docstring.
        """
        return prepare_inputs(source_nifti, density_nifti, workdir,
                                        **kwargs)

    @staticmethod
    def photopeak_cheatsheet(isotope=None):
        """Typical clinical energy windows for a known isotope — a quick
        reference to check BEFORE building anything, not tied to any
        specific simulation. Pulls straight from the builder section's ISOTOPES
        (the exact same registry build_simind_lu177/prepare_simind_input
        use for their own peak/collimator defaults), so this can never
        drift out of sync with what actually gets simulated.

        isotope: an ISOTOPES key (e.g. "lu177", "tc99m", "i131", "ac225"),
            case-insensitive with "-"/"_" ignored (same normalization
            prepare_simind_input's isotope= uses) — or None/"all" (default)
            for every known isotope at once.

        Returns, per isotope: {"isd", "collimator", "note", "peaks": [
            {"center_kev", "width_pct", "lo_keV", "hi_keV"}, ...]} — a dict
        keyed by isotope name if `isotope` was None/"all", or just that one
        isotope's dict directly if a specific name was given. lo_keV/hi_keV
        are computed with the exact same formula build_simind_lu177 uses
        for the .win files (center * (1 -+ width_pct/200)).

        Raises KeyError (via the builder section's isotope_config) for an unknown
        isotope name, listing what IS available.
        """
        def _one(name):
            cfg = isotope_config(name)
            peaks = []
            for center, width_pct in cfg["peaks"]:
                lo = center * (1 - width_pct / 200.0)
                hi = center * (1 + width_pct / 200.0)
                peaks.append({"center_kev": center, "width_pct": width_pct,
                              "lo_keV": round(lo, 2), "hi_keV": round(hi, 2)})
            return {"isd": cfg["isd"], "collimator": cfg["collimator"],
                   "note": cfg.get("note", ""), "peaks": peaks}

        if isotope is None or (isinstance(isotope, str) and isotope.lower() == "all"):
            return {name: _one(name) for name in ISOTOPES}
        return _one(isotope)

    @staticmethod
    def _load_debug_log(workdir):
        """Read `workdir`'s simulation-debug.json (or the newest
        build_log_*.json for older builds) directly, WITHOUT touching
        this file's module-level WORKDIR/BUILD_LOG globals
        the way that module's own load_build_log() does -- this has to stay
        safe to call for a workdir a reconstruct() call isn't currently
        using. Returns None if nothing readable is found."""
        fixed = os.path.join(workdir, "simulation-debug.json")
        path = fixed if os.path.exists(fixed) else None
        if path is None:
            cands = sorted(glob.glob(os.path.join(workdir, "build_log_*.json")))
            path = cands[-1] if cands else None
        if path is None:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def detect_isotope(workdir):
        """This build's isotope, for peak="auto"/peaks="auto". Prefers
        simulation-debug.json's recorded arguments.isotope (authoritative,
        works no matter how the folder is named), falling back to parsing
        it out of the folder name itself (parse_workdir's
        "<scanner>--<views>views--<isotope>--<spacing>mm" convention) for
        older builds without that log field. Returns None if neither source
        has it."""
        log = simind_v9._load_debug_log(workdir)
        if log is not None:
            iso = log.get("arguments", {}).get("isotope")
            if iso:
                return iso
        _, _, _, isotope, _ = simind_v9.parse_workdir(workdir)
        return isotope

    @staticmethod
    def auto_peaks(isotope):
        """Peak tag strings (e.g. ["113kev", "208kev"]) for every photopeak
        center photopeak_cheatsheet lists for this isotope -- bare tags, not
        {"center_kev":...} specs, so they match on FILE LOOKUP alone
        (resolve_peak skips window-width validation for plain strings) and
        can't be rejected just because the actual build used a non-default
        width. Used by reconstruct()'s peak="auto" and reconstruct_all()'s
        peaks="auto"."""
        sheet = simind_v9.photopeak_cheatsheet(isotope)
        return [f"{int(p['center_kev'])}kev" for p in sheet["peaks"]]

    def prepare_simind_input(self, source_nifti, density_nifti, workdir, *,
                              scanner=None, patient_id=None, **kwargs):
        """Build one SIMIND input set: .smc/.dmi/.smi/.win files, run_all.bat
        / run_all_mpi.bat, and a build_log_*.json provenance record, all
        written into `workdir`. This IS the builder section's build_simind_lu177 —
        a thin pass-through, not a reimplementation, so its full parameter
        list (n_azimuth, matrix_size, photon_multiplier_nn, source_hist_max,
        radius_of_rotation_cm, fov_axial_cm, tew_scatter, ...) is available
        via **kwargs; see that function's docstring for all of them.

        source_nifti / density_nifti should already be on the SPECT grid
        (i.e. the output of prepare_inputs()), not raw PET/CT.

        scanner:    overrides this instance's default scanner for this call.
        patient_id: segment tags become "{patient_id}_00", "{patient_id}_01",
                    ...; defaults to workdir's parent folder name if omitted
                    (see build_simind_lu177's docstring).

        Returns a BuildResult(workdir, segments, commands, win_file, log_file,
        windows). `windows` is {peak_keV: {"peak": {"lo_keV","hi_keV"},
        "lower": {...}, "upper": {...}}} — the energy window bounds actually
        written to each peak's .win file ("lower"/"upper" TEW scatter windows
        only present if tew_scatter=True). e.g. for a 113 keV Lu-177 peak:
        build.windows[113]["peak"] -> {"lo_keV": 101.7, "hi_keV": 124.3}.
        """
        scanner = scanner or self.scanner
        if not scanner:
            raise ValueError(
                "scanner is required — pass scanner=... here or set it in "
                "the simind_v9(scanner=...) constructor.")
        kwargs.setdefault("template_smc", self.template_smc)
        kwargs.setdefault("smc_dir", self.smc_dir)
        kwargs.setdefault("isotope", self.isotope)
        kwargs.setdefault("mpi_ranks", self.mpi_ranks)
        kwargs.setdefault("simind_bin_dir", self.simind_bin_dir)
        return build_simind_lu177(
            source_nifti=source_nifti, density_nifti=density_nifti,
            workdir=workdir, scanner=scanner, patient_id=patient_id, **kwargs)

    # ---------------------------------------------------------- reconstruct
    def reconstruct(self, simind_simulation_dir, peak="auto", *, beds=None,
                     build_log=None, activity_mbq=None, activity_mbq_mode=None,
                     time_per_proj_s=None,
                     n_iters=None, n_subsets=None, use_tew=None,
                     orient_projections_for_recon=None, flip_recon_axial=None,
                     convert_density_to_mu=None, mu_water=None,
                     algorithm=None, prior_type=None, prior_beta=None,
                     prior_gamma=None, relaxation_sequence=None,
                     calibration_factors=None, mu_waters=None):
        """Reconstruct every bed found in `simind_simulation_dir` — a
        "<patient_id>\\<scanner>--<views>--views" folder produced by
        prepare_simind_input() and then actually simulated (run_all_mpi.bat)
        — for one energy `peak`, stitching multi-bed results and writing
        registered NIfTI outputs back into that same folder. This IS
        the reconstruction section's run_reconstruction — a thin pass-through;
        see its docstring for exactly what each metric controls.

        simind_simulation_dir: the folder to reconstruct (== that module's
            WORKDIR). Must contain a build_log_*.json (from
            prepare_simind_input) plus the SIMIND Interfile output
            (.h00/.a00/.hct/.ict from actually running run_all_mpi.bat).
        peak: "auto" (default) — detect this build's isotope (see
            detect_isotope: from simulation-debug.json's recorded isotope,
            or the folder name) and reconstruct EVERY peak
            photopeak_cheatsheet lists for it (auto_peaks) — e.g. for a
            Lu-177 build this reconstructs 113+208 keV jointly (multi-peak,
            see below) with no peak= needed at all: just
            sim.reconstruct(workdir, algorithm="OSEM"). Raises ValueError if
            the isotope can't be determined (neither source has it) — pass
            peak= explicitly in that case.
            Otherwise: a single peak -- "113kev"/"208kev" tag string, or a
            {"center_kev":, "lower_pct":, "upper_pct":} spec (see
            the reconstruction section's resolve_peak) -- OR a LIST of either,
            for a JOINT multi-peak ("dual energy window") reconstruction
            combining every listed peak into ONE set of NC/AC/SC/ACSC images
            via PyTomography's ExtendedSystemMatrix (see
            reconstruct_bed_multi_peak / the dual-peak tutorial:
            https://pytomography.readthedocs.io/en/stable/notebooks/t_dualpeak.html).
            "auto" itself resolves to exactly this list form when an
            isotope has more than one typical peak.
        beds: explicit bed-tag list, or None (default) to auto-detect from
            the build log's segment tags, in build order.
        build_log: explicit build_log_*.json path, or None (default) to use
            the newest one found in simind_simulation_dir.
        activity_mbq, time_per_proj_s: scale the Poisson realization
            (statistics/noise level of the simulated acquisition).
        activity_mbq_mode: "per_bed" (default) applies activity_mbq to
            EVERY bed the same, only correcting their relative brightness
            (each bed acts like its own acquisition at that activity).
            "total_by_counts" instead treats activity_mbq as the TOTAL for
            the whole multi-bed study, split across beds by each bed's
            share of total simulated source counts — e.g.
            activity_mbq=10000 with a bed holding 20% of total counts uses
            2000 for that bed. See this file's
            segment_count_fraction docstring for exactly how the split is
            computed.
        n_iters, n_subsets: iteration/subset counts (n_subsets is ignored by
            MLEM and FBP -- see `algorithm`).
        use_tew: include lower/upper scatter windows (SC/ACSC variants) if
            the build has them.
        algorithm: one of "OSEM" (default), "MLEM", "OSMAPOSL", "BSREM",
            "RBIEM", "RBIMAP", "SART", "FBP" -- see
            https://pytomography.readthedocs.io/en/stable/notebooks/t_algorithms.html
            and this file's ALGORITHM config comment for
            what each one is and needs. OSMAPOSL/BSREM/RBIMAP REQUIRE
            prior_beta; RBIEM's prior is optional; the rest ignore priors.
        prior_type: "relative_difference" (default, RelativeDifferencePrior,
            uses prior_gamma too) or "quadratic" (QuadraticPrior, no gamma).
        prior_beta, prior_gamma: prior regularization strength / RDP's extra
            parameter. prior_beta=None (default) means no prior at all.
        relaxation_sequence: optional callable(n)->float overriding BSREM's
            or SART's own built-in relaxation schedule.
        calibration_factors, mu_waters: only used when `peak` is a list
            (multi-peak) -- per-peak lists matching its length. None ->
            calibration 1.0 for every peak (only physically correct without
            real cross-peak relative-efficiency data), mu_water from the
            built-in NIST table (113/208 keV only; required explicitly for
            any other peak).
        orient_projections_for_recon, flip_recon_axial: geometry-correction
            toggles — see run_reconstruction's/the module's docstring before
            changing these; the defaults are already verified correct for
            this pipeline.
        convert_density_to_mu, mu_water: attenuation-map unit handling
            (single-peak only; multi-peak uses mu_waters= per peak instead).

        Anything left as None (the default) falls back to
        this file's own current CONFIG-block value for that
        setting, for THIS CALL ONLY — no shared state leaks between repeated
        calls to this method (e.g. looping over several patients).

        Returns: {
            "workdir", "peak", "beds": [...],
            "per_bed": {bed: {"NC"/"AC"/"SC"/"ACSC": path, ...}, ...},
            "proj_per_bed": {bed: path, ...} for a single peak, or
                            {bed: {peak: path, ...}, ...} for multi-peak,
            "wholebody": {"NC"/"AC"/"SC"/"ACSC": path, ...},  # multi-bed only
            "proj_stitched": path or None,
        }
        """
        if isinstance(peak, str) and peak.lower() == "auto":
            isotope = self.detect_isotope(simind_simulation_dir)
            if not isotope:
                raise ValueError(
                    f"peak='auto' couldn't determine an isotope for "
                    f"{simind_simulation_dir} — not recorded in its "
                    f"simulation-debug.json, and not in the folder name "
                    f"either. Pass peak= explicitly instead.")
            resolved = self.auto_peaks(isotope)
            peak = resolved[0] if len(resolved) == 1 else resolved
            print(f"peak='auto' -> isotope={isotope!r} -> peak={peak}")
        return run_reconstruction(
            workdir=simind_simulation_dir, peak=peak, beds=beds,
            build_log=build_log, activity_mbq=activity_mbq,
            activity_mbq_mode=activity_mbq_mode,
            time_per_proj_s=time_per_proj_s, n_iters=n_iters,
            n_subsets=n_subsets, use_tew=use_tew,
            algorithm=algorithm, prior_type=prior_type, prior_beta=prior_beta,
            prior_gamma=prior_gamma, relaxation_sequence=relaxation_sequence,
            calibration_factors=calibration_factors, mu_waters=mu_waters,
            orient_projections_for_recon=orient_projections_for_recon,
            flip_recon_axial=flip_recon_axial,
            convert_density_to_mu=convert_density_to_mu, mu_water=mu_water)

    # --------------------------------------------------------- naming
    @staticmethod
    def build_workdir(root, patient_id, scanner, n_views, isotope,
                       resample_spacing_mm=4.0):
        """The one place the "<root>/<patient_id>/<scanner>--<views>views--
        <isotope>--<spacing>mm" folder convention is spelled out, so
        prepare-simulation.py and find_build_dirs()/parse_workdir() can
        never drift apart."""
        return os.path.join(
            root, patient_id,
            f"{scanner}--{n_views}views--{isotope}--{resample_spacing_mm:g}mm")

    @staticmethod
    def parse_workdir(workdir):
        """Inverse of build_workdir(): return (patient_id, scanner, n_views,
        isotope, resample_spacing_mm) parsed back out of a workdir path.
        scanner/n_views/isotope come back as None if the folder name doesn't
        match the convention at all (e.g. an old "<scanner>--<views>--views"
        folder from before isotope was added to the name); resample_spacing_mm
        alone comes back None if the folder predates ITS addition (isotope-
        only "<scanner>--<views>views--<isotope>" folders)."""
        workdir = workdir.rstrip("\\/")
        view_name = os.path.basename(workdir)
        patient_id = os.path.basename(os.path.dirname(workdir))
        parts = view_name.split("--")
        if len(parts) < 3:
            return patient_id, None, None, None, None
        scanner, views_part, isotope = parts[0], parts[1], parts[2]
        n_views = views_part[:-5] if views_part.lower().endswith("views") else views_part
        resample_spacing_mm = None
        if len(parts) >= 4:
            mm_part = parts[3]
            resample_spacing_mm = mm_part[:-2] if mm_part.lower().endswith("mm") else mm_part
        return patient_id, scanner, n_views, isotope, resample_spacing_mm

    # --------------------------------------------------------- batch/scan
    @staticmethod
    def _matches(value, allowed):
        """allowed == "all" (any case) matches everything; else `allowed` is
        a list compared to `value` case-insensitively."""
        if isinstance(allowed, str) and allowed.lower() == "all":
            return True
        return str(value).lower() in {str(a).lower() for a in allowed}

    @staticmethod
    def find_build_dirs(root, patients="all"):
        """Yield (patient_id, scanner, n_views, isotope, resample_spacing_mm,
        workdir) for every <root>/<patient_id>/<scanner>--<views>views--
        <isotope>--<spacing>mm folder that has a simulation-debug.json /
        build_log_*.json (i.e. was produced by prepare_simind_input).

        patients: "all" (default, case-insensitive) globs every patient
            directory under root, same as before. Or a list of patient IDs
            to restrict the scan to -- ONLY those patient directories get
            listed/globbed at all, instead of globbing root/*/* (every
            patient) and discarding the ones that don't match afterward.
            On a production-scale root (hundreds of patient folders on a
            shared/network filesystem) that root/*/* glob is the expensive
            part -- reconstruct_all()'s own patients= filter used to have
            no effect on this scan for exactly that reason (it only pruned
            AFTER find_build_dirs had already paid for globbing
            everything). Matching is case-insensitive, via the same
            _matches() used everywhere else in this class.
        """
        seen = set()
        try:
            all_patient_names = sorted(e.name for e in os.scandir(root) if e.is_dir())
        except FileNotFoundError:
            return
        patient_names = [p for p in all_patient_names if simind_v9._matches(p, patients)]
        for pattern in ("simulation-debug.json", "build_log_*.json"):
            for patient_name in patient_names:
                patient_dir = os.path.join(root, patient_name)
                for log_path in sorted(glob.glob(os.path.join(patient_dir, "*", pattern))):
                    workdir = os.path.dirname(log_path)
                    if workdir in seen:
                        continue
                    seen.add(workdir)
                    patient_id, scanner, n_views, isotope, resample_mm = \
                        simind_v9.parse_workdir(workdir)
                    if scanner is None:
                        continue    # doesn't match the naming convention
                    yield patient_id, scanner, n_views, isotope, resample_mm, workdir

    @staticmethod
    def has_simulation_output(workdir, peak, log=None):
        """Any bed simulated for this peak yet? Uses the same window_offsets
        -aware resolution as reconstruction (see the reconstruction section's 
        resolve_peak_files), so this works for BOTH:
          - combine_windows builds (any isotope with 2+ peaks -- lu177,
            ac225, ... -- one shared SIMIND run per bed covering ALL peaks;
            output files have no peak in their name at all, e.g.
            "NEMA_00_tot_w4.h00"), and
          - legacy / single-peak builds (one SIMIND run per peak; output
            files ARE named "..._{peak}_tot_w1.h00").
        log=None auto-loads simulation-debug.json (or the newest
        build_log_*.json) from workdir if present. (Case-insensitive on
        Windows automatically; SIMIND itself lowercases its own output
        filenames.)"""
        if log is None:
            log = simind_v9._load_debug_log(workdir)
        beds = [seg.get("tag") for seg in log.get("segments", [])] if log else []
        if not beds:
            # no log / no segments -- fall back to the original glob, which
            # only ever matched the legacy per-peak naming anyway.
            return len(glob.glob(os.path.join(workdir, f"*_{peak}_tot_w1.h00"))) > 0
        for bed in beds:
            base, win = resolve_peak_files(bed, peak, log=log,
                                                   workdir=workdir)
            if os.path.exists(f"{base}_tot_w{win['peak']}.h00"):
                return True
        return False

    @staticmethod
    def _resolve_per_isotope(value, isotope, name):
        """value: either a plain scalar (returned unchanged -- the original,
        one-value-for-everything behavior) or a dict {isotope: value},
        looked up by `isotope` case-insensitively. Lets reconstruct_all()
        take per-isotope activity_mbq/time_per_proj_s across a batch
        spanning several tracers (e.g. Ac-225's alpha-therapy activity is
        nothing like Lu-177's, which is nothing like a Tc-99m diagnostic
        scan) instead of forcing one fixed value onto every isotope in the
        batch. Raises a clear error if `isotope` has no entry, rather than
        silently reconstructing with the wrong number."""
        if not isinstance(value, dict):
            return value
        for k, v in value.items():
            if str(k).lower() == str(isotope).lower():
                return v
        raise ValueError(
            f"{name} was given as a dict {value!r} but has no entry for "
            f"isotope {isotope!r} — add one (keys are case-insensitive), "
            f"or pass a plain number to use the same {name} for every "
            f"isotope.")

    @staticmethod
    def _resolved_recon_dir(workdir, reconstruct_kwargs):
        """The exact recon_dir a reconstruct(**reconstruct_kwargs) call
        would write into, computed WITHOUT reconstructing anything — so
        already_reconstructed() can check the matching parameter-named
        subfolder specifically. Each osem parameter combo gets its own
        subfolder (see the reconstruction section's recon_output_dir), so
        "already done" has to mean "already done with THESE settings", not
        just "a recon exists somewhere" — otherwise reconstructing the same
        build again with different settings would be skipped by mistake."""
        algorithm = reconstruct_kwargs.get("algorithm") or ALGORITHM
        n_iters = reconstruct_kwargs.get("n_iters") or N_ITERS
        n_subsets = reconstruct_kwargs.get("n_subsets") or N_SUBSETS
        time_per_proj_s = reconstruct_kwargs.get("time_per_proj_s") or TIME_PER_PROJ_S
        activity_mbq = reconstruct_kwargs.get("activity_mbq") or ACTIVITY_MBQ
        return recon_output_dir(workdir, algorithm, n_iters, n_subsets,
                                       time_per_proj_s, activity_mbq)

    @staticmethod
    def already_reconstructed(workdir, peak, **reconstruct_kwargs):
        """Looks like reconstruct() already ran for this peak WITH THESE
        EXACT settings (recon-debug.json inside the matching parameter
        subfolder already has an entry for this peak)."""
        recon_dir = simind_v9._resolved_recon_dir(workdir, reconstruct_kwargs)
        debug_path = os.path.join(recon_dir, "recon-debug.json")
        if not os.path.exists(debug_path):
            return False
        try:
            with open(debug_path, "r", encoding="utf-8") as f:
                debug_log = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        return peak in debug_log.get("runs", {})

    def reconstruct_all(self, root, *, patients="all", scanners="all",
                         views="all", isotopes="all", resample_mm="all",
                         peaks="auto",
                         skip_done=True, **reconstruct_kwargs):
        """Find every simulation build under `root` (see find_build_dirs),
        filter by patient/scanner/view-count/isotope/resample-spacing, and
        reconstruct() each one for every peak in `peaks` — skipping a
        build/peak that has no simulation output yet (.h00 missing —
        run_all_mpi.bat hasn't been run), and, if skip_done, one already
        reconstructed with these exact settings. This is the whole loop
        behind reconstruct-simulations.py; calling code just configures and
        calls this once.

        root: folder to scan, e.g. r"C:\\simind\\v9\\patients".
        patients, scanners, views, isotopes, resample_mm: "all" (default,
            case-insensitive) or a list to filter to, e.g.
            patients=["psma_000000", "NEMA"].
        peaks: "auto" (default) — for EACH build, detect its isotope
            (detect_isotope) and reconstruct every peak photopeak_cheatsheet
            lists for it (auto_peaks), independently (one reconstruct()
            call per peak — same as passing that isotope's peak list
            explicitly; NOT a joint multi-peak combination — pass an
            explicit list of peaks as a single reconstruct() call's peak=
            for that, see its docstring). A build whose isotope can't be
            determined is skipped (counted under skipped_filter) rather
            than failing the whole scan.
            Otherwise: an explicit list of peaks to reconstruct on EVERY
            matched build regardless of isotope — each either a tag string
            (e.g. "113kev") or a {"center_kev":, "lower_pct":, "upper_pct":}
            spec (see the reconstruction section's resolve_peak).
        skip_done: skip a build/peak already reconstructed with the SAME
            n_iters/n_subsets/time_per_proj_s/activity_mbq (each parameter
            combination gets its own output subfolder, so a different
            combination is never mistaken for "already done"). False forces
            re-reconstructing everything matched regardless.
        **reconstruct_kwargs: forwarded to reconstruct() on every call
            (n_iters, n_subsets, use_tew, activity_mbq, time_per_proj_s, ...).
            activity_mbq and time_per_proj_s may each be EITHER a plain
            number (used for every build, as before) OR a dict keyed by
            isotope, e.g. activity_mbq={"lu177": 750, "ac225": 40,
            "tc99m": 500} — looked up (case-insensitively) per build using
            that build's isotope, so a batch spanning several tracers with
            very different realistic activities/dwell-times doesn't have to
            share one fixed value. A dict missing an isotope present in the
            scan raises a clear error rather than guessing.

        Prints progress as it goes. Returns a summary dict: {"ran",
        "skipped_filter", "skipped_no_sim", "skipped_done", "failed": counts,
        "results": [the reconstruct() return dict for each run that happened]}.
        """
        auto = isinstance(peaks, str) and peaks.lower() == "auto"
        # patients= is passed through here so find_build_dirs can skip
        # globbing non-matching patient directories entirely (see its own
        # docstring) -- scanners/views/isotopes/resample_mm still filter
        # AFTER this, in the loop below, since those aren't known until a
        # build's workdir name is parsed.
        builds = list(self.find_build_dirs(root, patients=patients))
        print(f"found {len(builds)} build folder(s) under {root}")
        print(f"  patients={patients}  scanners={scanners}  views={views}  "
              f"isotopes={isotopes}  resample_mm={resample_mm}  "
              f"peaks={'auto (per-build isotope)' if auto else list(peaks)}  "
              f"skip_done={skip_done}\n")

        summary = {"ran": 0, "skipped_filter": 0, "skipped_no_sim": 0,
                   "skipped_done": 0, "failed": 0, "results": []}

        for patient_id, scanner, n_views, isotope, spacing, workdir in builds:
            tag = f"{patient_id} / {scanner}--{n_views}views--{isotope}--{spacing}mm"

            if not (self._matches(patient_id, patients)
                    and self._matches(scanner, scanners)
                    and self._matches(n_views, views)
                    and self._matches(isotope, isotopes)
                    and self._matches(spacing, resample_mm)):
                print(f"[skip-filter]  {tag}")
                summary["skipped_filter"] += 1
                continue

            build_log = self._load_debug_log(workdir)

            # resolve any dict-by-isotope activity_mbq/time_per_proj_s (see
            # docstring) down to a plain number for THIS build, before
            # either the already-done check or the actual reconstruct()
            # call -- both need to agree on the same resolved value.
            build_kwargs = dict(reconstruct_kwargs)
            for key in ("activity_mbq", "time_per_proj_s"):
                if key in build_kwargs:
                    build_kwargs[key] = self._resolve_per_isotope(
                        build_kwargs[key], isotope, key)

            if auto:
                build_isotope = self.detect_isotope(workdir)
                if not build_isotope:
                    print(f"[skip-filter]  {tag}  (peaks='auto' but isotope "
                          f"couldn't be determined)")
                    summary["skipped_filter"] += 1
                    continue
                build_peaks = self.auto_peaks(build_isotope)
                print(f"  {tag}: isotope={build_isotope!r} -> "
                      f"peaks={build_peaks}")
            else:
                build_peaks = peaks

            for peak in build_peaks:
                peak_label = peak if isinstance(peak, str) else (
                    f"{peak['center_kev']:g}kev"
                    f"+{peak.get('upper_pct', peak.get('width_pct', 10) / 2):g}%"
                    f"-{peak.get('lower_pct', peak.get('width_pct', 10) / 2):g}%")
                peak_tag = f"{tag} @ {peak_label}"
                # only the bare "{center}kev" tag is a real filename to
                # check for on disk; a dict spec's actual file lookup tag
                # is resolved (and validated) inside reconstruct() itself.
                file_tag = peak if isinstance(peak, str) else f"{int(peak['center_kev'])}kev"

                if not self.has_simulation_output(workdir, file_tag, log=build_log):
                    print(f"[skip-no-sim]  {peak_tag}  (no .h00 yet — run "
                          f"run_all_mpi.bat / run_all_patients.bat first)")
                    summary["skipped_no_sim"] += 1
                    continue

                if skip_done and self.already_reconstructed(
                        workdir, file_tag, **build_kwargs):
                    print(f"[skip-done]    {peak_tag}  "
                          f"(already reconstructed with these settings)")
                    summary["skipped_done"] += 1
                    continue

                print(f"[reconstructing] {peak_tag} ...")
                try:
                    result = self.reconstruct(workdir, peak=peak,
                                              **build_kwargs)
                    out = result["wholebody"] or next(
                        iter(result["per_bed"].values()), {})
                    print(f"  done: {out}")
                    summary["ran"] += 1
                    summary["results"].append(result)
                except FileNotFoundError as e:
                    # e.g. amap missing even though photopeak .h00 exists
                    print(f"  [skip-no-sim]  incomplete simulation output: {e}")
                    summary["skipped_no_sim"] += 1
                except Exception as e:
                    print(f"  FAILED: {e}")
                    summary["failed"] += 1

        print("\n" + "=" * 60)
        print(f" done.  ran={summary['ran']}  "
              f"skipped-filter={summary['skipped_filter']}  "
              f"skipped-no-sim={summary['skipped_no_sim']}  "
              f"skipped-done={summary['skipped_done']}  "
              f"failed={summary['failed']}")
        print("=" * 60)
        return summary


if __name__ == "__main__":
    sim = simind_v9(scanner="siemens_symbia_t")
    print(sim)
