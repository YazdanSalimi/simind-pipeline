"""
Build SIMIND inputs for one or more patients, sweeping isotopes, resample
spacings, scanners, and view counts. Verified on both Windows and Linux.
Produces run_all.sh/run_all.bat/run_all_mpi.bat per <patient_id>\\<scanner>--
<views>views--<isotope>--<spacing>mm folder, ready to run directly (each is
a plain-text script -- one `simind`/`simind_mpi` command per line, so
parameters like the /NN: history multiplier or MPI rank count can be
hand-edited without rebuilding the dataset), and then reconstruct-
simulations.py to reconstruct them.

Based on run_test_v9.py, but goes through the simind_yazdan.simind_v9 class
(prepare_inputs / prepare_simind_input) instead of calling
simind_lu177_v9 directly, so this and reconstruct-simulations.py always
track the same tested, bug-fixed pipeline.

PARALLEL: runs in two pool-parallel phases instead of one big serial loop
(pool_execute-style, adapted from the user's own multiprocessing.Pool +
tqdm helper) --
  Phase 1: resample each (patient, spacing) pair's PET/CT ONCE, in
           parallel -- this is the expensive step (full-res image I/O +
           resampling), and every isotope x scanner x view combo below
           reuses its result, so it must finish first but only needs to
           run once per (patient, spacing), not once per combo.
  Phase 2: build every isotope x scanner x view combo (the .smc/.dmi/.smi/
           .win writing), in parallel -- this is where the bulk of the
           combos live (isotopes x scanners x views per patient x spacing),
           so it's the main parallelism win.
Uses multiprocessing's "spawn" start method (required for correctness on
Windows/macOS -- each worker re-imports this file fresh, which is why the
actual pool dispatch code below is guarded by `if __name__ == "__main__":`
-- everything ABOVE that guard, including the CONFIG block, re-runs safely
in every worker process on import; only the dispatch itself must not).

Edit the config block and run:  python prepare-simulation.py
"""
#%%
from simind_yazdan import simind_v9
import os
import itertools
import multiprocessing
import concurrent.futures as cf
from tqdm import tqdm
from natsort import os_sorted
from glob import glob
#%%
# ============================== config =================================
simulation_root = "/path/to/simulation-place"                    # output root
template = "/path/to/simind/v9/simind.smc"                       # any valid v9 .smc as base
smc_dir = "/path/to/simind/v9/smc_dir"                            # where the .isd spectra live
scanners = ["siemens_symbia_t", "siemens_intevo", "siemens_ecam", "ge_nm870", "ge_discovery_670", "ge_infinia", "ge_starguide", "philips_brightview", "mediso_anyscan", "generic_nai"]                # see SCANNER_DB in simind_lu177_v9.py

# every isotope IS built at every resample spacing -- a full cross product
# (isotopes x resample_spacings_mm), not a 1:1 pairing. Sizes don't need to
# match.
isotopes = ["lu177", "tc99m", "ac225"]          # lu177 / tc99m / i131 / ac225
resample_spacings_mm = [2.0, 3.0, 4.0, 4.8]

density_is_hu = True                            # CT is in Hounsfield units
# "linear" (default, blends across edges -- a phantom's sharp-edged sphere
# gets a ramped boundary instead of a clean step), "nearest" (exact input
# values kept, stepped/aliased boundary -- use for quantitative phantom
# validation), or "bspline" (smoother than linear, costs more compute).
# Independent per modality.
pet_interp = "linear"
ct_interp = "linear"
view_counts = [64, 96, 128]     # e.g. [16, 32, 64, 128]
nn = 1                         # /NN history multiplier at runtime -- dropped from 10
                                # (2026-08-XX): runtime is ~linear in total histories, and
                                # NN=10 risked not finishing within shared-cpu's 12h limit,
                                # which (see run_all_server.sh's claim mechanism) means a
                                # killed job leaves a stale .simind_claim AND has no per-bed
                                # resume -- the whole combo reruns from scratch next time.
                                # NN=1 is the SIMIND manual's own stated safe default (only
                                # NN<1.0 is called out as risky -- some voxels could get
                                # skipped by Monte Carlo rounding); this just accepts more
                                # noise, especially in low-activity voxels, for a ~10x cut
                                # in runtime and reliably finishing within one segment.
source_hist_max = 10000         # brightest source voxel emits this many histories
mpi_ranks = 28                  # ranks for run_all_mpi.bat
# folder holding the `simind` executable on the machine that will actually
# RUN these builds (e.g. a Linux/HPC install) -- if set, every generated
# run_all.sh gets an `export PATH="<this>:$PATH"` line prepended, so it can
# be run standalone from anywhere (`bash /abs/path/to/run_all.sh`) without
# needing `simind` already on PATH first. None (default) leaves run_all.sh
# unchanged. Only affects the .sh script; run_all.bat/run_all_mpi.bat
# (Windows) are unaffected.
simind_bin_dir = "/path/to/simind/v9"
# simind_bin_dir = None
# SMC Flag-1: verbose on-line printout + results to the terminal during
# the run. SIMIND's own manual says to turn this OFF for batch-queue
# submissions -- False (default here) does that. Set True for a small
# interactive test build where you want to watch live output. NOTE: this
# is baked into the .smc at BUILD time, so it only affects builds made
# AFTER changing it -- for already-built folders, append /fa:1 to the
# `simind`/`simind_mpi` command instead (no rebuild needed; run_all_server.sh
# does this automatically via its own SUPPRESS_SIMIND_PRINTOUT setting).
on_line_printout = False

# how many worker PROCESSES to build with -- -1 (default) uses every CPU
# core. BOTH phases hold a full-res PET/CT volume (or two) per worker at
# once (Phase 1 resamples; Phase 2's build_simind_lu177 re-reads the
# already-prepared volumes) -- num_workers x one-item's peak RSS must fit
# inside whatever memory your salloc/sbatch actually requested, or workers
# start getting OOM-killed under load (usually only once several are
# running concurrently -- see _pool_execute's crash-isolation retry logic
# for what happens if that occurs mid-run). Measure one item's real peak
# RSS with `/usr/bin/time -v` before picking a number for a big salloc
# rather than guessing.
# NUM_WORKERS_OVERRIDE env var wins over this if set (so you can retune
# per-salloc without editing this file each time), e.g.:
#   NUM_WORKERS_OVERRIDE=16 python prepare-simulation.py
num_workers = int(os.environ.get("NUM_WORKERS_OVERRIDE", -1))

# ---- source PET/CT images to build -------------------------------------
source = "./scratch/spect-simind-simulations/spect-simulation/petct-data"
list_pet_images = os_sorted(glob(os.path.join(source, "PET--crop", "*.nii.gz")))[:1]
list_pet_images = [x for x in list_pet_images if not "--body-contour.nii.gz" in x]
list_ct_images = [os.path.join(source, "CT--crop-matched-to-pet", os.path.basename(x))
                   for x in list_pet_images]
#%%
# ------------------------------------------------------------------------
# isotope is intentionally NOT set here -- it's passed explicitly per call
# below instead (prepare_simind_input's isotope= kwarg always wins over
# this instance's default), since one `sim` covers every isotope in the
# sweep; only smc_dir/template/mpi_ranks/simind_bin_dir are shared config.
# Re-created fresh in every worker process too (spawn re-imports this
# module top-to-bottom before running any worker function), so each
# worker's own `sim` is always valid without needing to pickle/share one
# across process boundaries.
sim = simind_v9(smc_dir=smc_dir, template_smc=template,
                 mpi_ranks=mpi_ranks, simind_bin_dir=simind_bin_dir)


def _pool_execute(map_function, list_inputs, num_workers=-1,
                  desc="Doing task in parallel", colour="green", ncols=95,
                  max_retries=3):
    """ProcessPoolExecutor + tqdm helper (adapted from the user's own
    pool_execute, which used multiprocessing.Pool's imap_unordered --
    switched to concurrent.futures.ProcessPoolExecutor here instead, see
    why below). Work completes out of order internally, but the returned
    list is realigned to match list_inputs' own order (results[i] always
    corresponds to list_inputs[i]) -- unlike imap_unordered, so callers can
    zip() it back against their own input list directly. A None entry
    means that input's worker crashed repeatedly and was given up on (see
    max_retries below) -- callers must check for it. "spawn" context is
    required for correctness on Windows/macOS.

    WHY ProcessPoolExecutor, NOT multiprocessing.Pool: if a worker process
    dies abruptly (segfault in a native lib like ITK/SimpleITK, OOM-kill,
    etc. -- not a catchable Python exception), plain Pool.imap_unordered
    never notices -- the task that worker was holding is orphaned forever
    and the whole call just hangs, silently, often right near the end
    (whichever tasks happen to still be in flight when it dies are the
    ones lost). ProcessPoolExecutor DOES notice: every pending/future
    submission raises BrokenProcessPool as soon as it happens, so we can
    catch it, figure out which inputs are still unresolved, and retry just
    those (with fewer workers each retry, to isolate a chronically-crashy
    input rather than let it keep taking down its concurrent siblings)
    instead of hanging forever.
    """
    list_inputs = list(list_inputs)
    n = len(list_inputs)
    if n == 0:
        print(f"{desc}: empty list given, nothing to do")
        return []
    max_workers = min(n, os.cpu_count() or 1)
    if num_workers == -1:
        num_workers = max_workers
    else:
        num_workers = max(1, min(num_workers, max_workers))

    results = [None] * n
    pending = list(range(n))
    attempt = 0
    n_crashes = 0
    ctx = multiprocessing.get_context("spawn")
    while pending:
        attempt += 1
        this_round, pending = pending, []
        # first pass runs at full parallelism; ANY crash drops straight to
        # serial (1 worker) for all further rounds -- with only one task in
        # flight at a time, submission order == execution order (FIFO), so
        # once a crash happens the FIRST still-unresolved item is provably
        # the one that killed its worker (nothing else was running
        # concurrently to blame instead). That lets us drop exactly that
        # one item and keep making guaranteed progress on the rest, instead
        # of re-submitting the whole batch together and hitting the same
        # crash again before any of its innocent neighbors get a turn.
        workers_this_round = num_workers if attempt == 1 else 1
        label = f"{desc} with {workers_this_round} workers!"
        if attempt > 1:
            label += f" (serial retry, isolating crashing item)"
        with cf.ProcessPoolExecutor(max_workers=workers_this_round, mp_context=ctx) as ex:
            futures = {ex.submit(map_function, list_inputs[i]): i for i in this_round}
            pbar = tqdm(total=len(this_round), colour=colour, ncols=ncols, desc=label)
            resolved = set()
            try:
                for fut in cf.as_completed(futures):
                    i = futures[fut]
                    results[i] = fut.result()
                    resolved.add(i)
                    pbar.update(1)
            except cf.process.BrokenProcessPool:
                # NOTE: once the pool breaks, EVERY pending future is
                # immediately marked .done() too (with the broken-pool
                # exception attached) -- so f.done() can't tell a real
                # result apart from a casualty. Use `resolved` (only ever
                # populated by an actual successful fut.result() above)
                # instead.
                unresolved = [i for i in this_round if i not in resolved]
                pbar.close()
                n_crashes += 1
                if workers_this_round == 1:
                    bad_i = unresolved[0]
                    results[bad_i] = None
                    print(f"  !! item {bad_i} of '{desc}' crashed its worker "
                          f"process (segfault/OOM?) -- giving up on it, "
                          f"continuing with the other {len(unresolved) - 1}")
                    pending = unresolved[1:]
                else:
                    print(f"  !! a worker process died unexpectedly "
                          f"(segfault/OOM?) during '{desc}' -- {len(unresolved)} "
                          f"item(s) unresolved, switching to serial retry "
                          f"to isolate which one")
                    pending = unresolved
                if n_crashes > max_retries and pending:
                    for i in pending:
                        results[i] = None
                    print(f"  !! {n_crashes} worker crashes during '{desc}' -- "
                          f"giving up on the remaining {len(pending)} item(s) "
                          f"(looks systemic, not a single bad input)")
                    pending = []
                continue
            pbar.close()
    return results


def _prepare_one(args):
    """Phase 1 worker: resample ONE (patient, spacing) pair's PET/CT once.
    Returns (patient_id, resample_spacing_mm, src_path, den_path, error_or_None).
    """
    patient_id, pet_url, ct_url, resample_spacing_mm = args
    prep = os.path.join(simulation_root, patient_id,
                        f"prep--{resample_spacing_mm:g}mm")
    try:
        src, den = sim.prepare_inputs(pet_url, ct_url, prep,
                                      density_is_hu=density_is_hu,
                                      resample_spacing_mm=resample_spacing_mm,
                                      pet_interp=pet_interp, ct_interp=ct_interp)
        return (patient_id, resample_spacing_mm, src, den, None)
    except Exception as e:
        return (patient_id, resample_spacing_mm, None, None, repr(e))


def _build_one(args):
    """Phase 2 worker: build ONE isotope x scanner x view combo, given the
    ALREADY-prepared src/den for its (patient, spacing) from Phase 1.
    Returns (ok, patient_id, scanner, nv, isotope, resample_spacing_mm,
    workdir_or_None, bed_tags_or_None, windows_or_None, error_or_None).
    """
    patient_id, resample_spacing_mm, src, den, isotope, scanner, nv = args
    try:
        workdir = sim.build_workdir(simulation_root, patient_id, scanner,
                                    nv, isotope, resample_spacing_mm)
        build = sim.prepare_simind_input(
            source_nifti=src,
            density_nifti=den,
            workdir=workdir,
            scanner=scanner,
            isotope=isotope,
            n_azimuth=nv,
            matrix_size=None,
            auto_matrix=True,
            photon_multiplier_nn=nn,
            source_hist_max=source_hist_max,
            tew_scatter=True,
            on_line_printout=on_line_printout,
            patient_id=patient_id,
        )
        return (True, patient_id, scanner, nv, isotope, resample_spacing_mm,
                workdir, [s["tag"] for s in build.segments], list(build.windows), None)
    except Exception as e:
        return (False, patient_id, scanner, nv, isotope, resample_spacing_mm,
                None, None, None, repr(e))


if __name__ == "__main__":
    # ---- Phase 1: resample every (patient, spacing) pair once, in parallel
    phase1_inputs = [
        (os.path.basename(pet_url).replace(".nii.gz", ""), pet_url, ct_url, spacing)
        for pet_url, ct_url in zip(list_pet_images, list_ct_images)
        for spacing in resample_spacings_mm
    ]
    phase1_results = _pool_execute(_prepare_one, phase1_inputs,
                                   num_workers=num_workers,
                                   desc="Phase 1/2: resampling PET/CT")

    prepared = {}   # (patient_id, spacing) -> (src, den)
    prep_ok = prep_failed = 0
    for idx, result in enumerate(phase1_results):
        if result is None:
            # worker crashed repeatedly (see _pool_execute) -- identify which
            # input this was from phase1_inputs (same index/order)
            patient_id, _, _, spacing = phase1_inputs[idx]
            print(f"  !! FAILED prepare_inputs {patient_id} @ {spacing:g}mm: "
                  f"worker process crashed repeatedly (segfault/OOM?)")
            prep_failed += 1
            continue
        patient_id, spacing, src, den, err = result
        if err is not None:
            print(f"  !! FAILED prepare_inputs {patient_id} @ {spacing:g}mm: {err}")
            prep_failed += 1
            continue
        prepared[(patient_id, spacing)] = (src, den)
        prep_ok += 1

    # ---- Phase 2: build every isotope x scanner x view combo, in parallel
    phase2_inputs = [
        (patient_id, spacing, src, den, isotope, scanner, nv)
        for (patient_id, spacing), (src, den) in prepared.items()
        for isotope, scanner, nv in itertools.product(isotopes, scanners, view_counts)
    ]
    phase2_results = _pool_execute(_build_one, phase2_inputs,
                                   num_workers=num_workers,
                                   desc="Phase 2/2: building SIMIND inputs")

    built = failed = 0
    for idx, result in enumerate(phase2_results):
        if result is None:
            # worker crashed repeatedly (see _pool_execute)
            patient_id, spacing, _, _, isotope, scanner, nv = phase2_inputs[idx]
            print(f"  !! FAILED {patient_id} / {scanner} / {nv}views / "
                  f"{isotope} / {spacing:g}mm: worker process crashed "
                  f"repeatedly (segfault/OOM?)")
            failed += 1
            continue
        ok, patient_id, scanner, nv, isotope, spacing, workdir, tags, windows, err = result
        if ok:
            built += 1
            print(f"=== {patient_id} / {scanner} / {nv}views / {isotope} / "
                  f"{spacing:g}mm -> {workdir}")
            print(f"    beds={tags}  windows(keV)={windows}")
        else:
            failed += 1
            print(f"  !! FAILED {patient_id} / {scanner} / {nv}views / "
                  f"{isotope} / {spacing:g}mm: {err}")

    print(f"""
Done building: {built} ok, {failed} failed (of {len(phase2_inputs)} combos);
Phase 1 (resampling): {prep_ok} ok, {prep_failed} failed (of {len(phase1_inputs)}).

To run every simulation, run each combo folder's own script (Windows:
run_all.bat / run_all_mpi.bat, Linux: run_all.sh) -- these are plain text,
so simulation parameters (e.g. /NN: history multiplier, MPI rank count) can
be edited per run without rebuilding the .smc/.dmi/.smi inputs.

Then reconstruct:
    python reconstruct-simulations.py
""")
