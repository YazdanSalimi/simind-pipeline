"""
Reconstruct every SIMIND simulation build found under a root folder
(default C:\\simind\\v9\\patients) for one or more energy peaks, skipping
anything already reconstructed. Runs after the Monte Carlo simulation step
(run_all.sh / run_all.bat / run_all_mpi.bat per build); this script does
the reconstruction step.

All the scanning/filtering/skip-logic/looping lives in simind_yazdan.simind_v9
itself (reconstruct_all) — this file is just configuration and one call.

Edit the config block and run:  python reconstruct-simulations.py
"""
from simind_yazdan import simind_v9

# ============================== config ================================
root = r"C:\simind\v9\spect-simulation"

# "all" (case-insensitive) or a list, e.g. ["psma_000000", "NEMA"]
patients = "all"
scanners = "all"
views = "all"                      # view counts, as strings, e.g. ["16", "64"]
isotopes = "all"                   # e.g. ["lu177"]
resample_mm = "all"                # e.g. ["4"] -- the resample_spacing_mm
                                    # a build used, as strings

# "auto" (default): for EACH build, detect its isotope (from
# simulation-debug.json, or the folder name) and reconstruct every peak
# photopeak_cheatsheet lists for it -- independently, one reconstruct() call
# per peak (e.g. a Lu-177 build gets 113kev and 208kev separately; a build
# whose isotope can't be determined is skipped, not failed). No need to
# know ahead of time which isotopes are under `root` -- it can span several.
#
# To pin an explicit peak list instead (applied to EVERY matched build
# regardless of isotope), set peaks to a list. Each entry is either a tag
# string matching what was simulated (e.g. "113kev"), or a {"center_kev":,
# "lower_pct":, "upper_pct":} spec -- the spec is checked against what was
# ACTUALLY simulated for that center and errors clearly if it doesn't match
# (SIMIND can't be re-windowed after the fact; see
# reconstruct_lu177_multibed.resolve_peak). A spec only works if a build
# actually simulated that exact window (e.g. energy_windows=[(113, 10),
# (208, 10)] in prepare-simulation.py for +5%/-5% on both of Lu-177's peaks).
peaks = "auto"
# peaks = [
#     {"center_kev": 113, "lower_pct": 5, "upper_pct": 5},
#     {"center_kev": 208, "lower_pct": 5, "upper_pct": 5},
#     {"center_kev": 218, "lower_pct": 5, "upper_pct": 5},
# ]

n_iters = 4
n_subsets = 8
use_tew = True
# activity_mbq / time_per_proj_s: EITHER a plain number (used for every
# build regardless of isotope, as below) OR a dict keyed by isotope,
# looked up per build (case-insensitive) -- for a batch spanning several
# tracers with very different realistic activities/dwell-times (e.g.
# Ac-225 alpha-therapy activity is nothing like Lu-177's, which is nothing
# like a Tc-99m diagnostic scan). A dict missing an isotope present in the
# scan raises a clear error instead of guessing. Example:
activity_mbq = {"lu177": 750, "ac225": 400, "tc99m": 500}
time_per_proj_s = {"lu177": 15.0, "ac225": 60.0, "tc99m": 100.0}
# activity_mbq = 7.5e+2
# time_per_proj_s = 15.0             
activity_mbq_mode = "total_by_counts" # total_by_counts or per_bed

# "OSEM" (default) / "MLEM" / "OSMAPOSL" / "BSREM" / "RBIEM" / "RBIMAP" /
# "SART" -- see reconstruct_lu177_multibed.py's ALGORITHM config comment for
# what each needs. OSMAPOSL/BSREM/RBIMAP REQUIRE prior_beta; RBIEM's prior is
# optional; OSEM/MLEM/SART ignore prior_* entirely.
algorithm = "OSEM"
prior_type = "relative_difference"   # or "quadratic" (no prior_gamma)
prior_beta = None                    # None = no prior
prior_gamma = 1.0
relaxation_sequence = None           # BSREM/SART only; None = library default

# only used when `peaks` above resolves to more than one peak (joint multi-
# peak reconstruction) -- must match the peak count, or None for defaults
# (calibration 1.0 each, built-in NIST mu_water per peak).
calibration_factors = None
mu_waters = None

# skip a (build, peak) already reconstructed with these exact settings (each
# n_iters/n_subsets/time_per_proj_s/activity_mbq/algorithm combination gets
# its own output subfolder). Set False to force re-reconstructing everything
# matched.
skip_done = True
# ========================================================================

if __name__ == "__main__":
    sim = simind_v9()
    sim.reconstruct_all(
        root, patients=patients, scanners=scanners, views=views,
        isotopes=isotopes, resample_mm=resample_mm, peaks=peaks,
        skip_done=skip_done,
        n_iters=n_iters, n_subsets=n_subsets, use_tew=use_tew,
        activity_mbq=activity_mbq, time_per_proj_s=time_per_proj_s,
        activity_mbq_mode=activity_mbq_mode,
        algorithm=algorithm, prior_type=prior_type, prior_beta=prior_beta,
        prior_gamma=prior_gamma, relaxation_sequence=relaxation_sequence,
        calibration_factors=calibration_factors, mu_waters=mu_waters)
