"""Run a training script on Kaggle's free GPU, from this Mac, in one command.

    python tools/kaggle_run.py inflow/futuretst_v2.py

What it does:

  1. bundles the code and the ~5 MB of arrays the ML stages need into a private
     Kaggle Dataset (created the first time, versioned after that);
  2. pushes a GPU kernel that runs your script against that dataset;
  3. polls until the run finishes;
  4. downloads whatever the script wrote back into results/ and checkpoints/.

Only the ML stages are supported, and that is deliberate: the VIC branch needs a
compiled Fortran/C binary and 1.6 GB of forcing, which is not worth shipping.

ONE-TIME SETUP
  pip install kaggle
  Kaggle -> your profile -> Settings -> API -> "Create New Token".
  Save the downloaded kaggle.json to ~/.kaggle/kaggle.json, then:
      chmod 600 ~/.kaggle/kaggle.json
  Put your Kaggle username in KAGGLE_USER below (or export KAGGLE_USERNAME).

WHAT TO EXPECT
  Kaggle gives roughly 30 GPU-hours per week, and a single session is capped at
  about 12 hours.  Both models here are small -- the field CNN is a 539 KB
  checkpoint, FutureTST v2 is d_model 64 with 2+2 layers -- so expect a useful
  but not enormous speed-up over this Mac, and the larger practical win is that
  your laptop stays free and several experiments can run at once.

  The kernel runs with internet DISABLED, which is what you want: every input
  arrives through the dataset, so a run is reproducible and cannot silently
  depend on something downloaded at runtime.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KAGGLE_USER = os.environ.get("KAGGLE_USERNAME", "")
DATASET_SLUG = "streamflow-hirakud"
KERNEL_SLUG = "streamflow-run"

# Code the ML stages import.  preprocessing/ is included only for the few
# modules cnn/ and inflow/ import at module level.
CODE_DIRS = ["configs", "cnn", "inflow", "preprocessing", "hydrology"]

# The ~5 MB the ML stages actually read.  Everything else stays on the Mac.
DATA_FILES = [
    "coarse_grid.npz", "fine_target.npz", "dem_fine.npz", "climatology.npz",
    "daily_fine_obs.npz", "rain_thresholds.npz", "feature_scaler.npz",
    "coord_scaler.npz", "basin.npz",
    # forecast_catchment_met.parquet (v1, 2004-2014, leads 1-17) was archived
    # 2026-09-07 -- superseded by forecast_catchment_met_v2.parquet below.
    "daily_catchment_met.parquet",
    "daily_catchment_obs.parquet", "inflow_daily.parquet",
    "inflow_dataset.parquet", "catchment_rainfall.parquet",
    "loyo_oof_rainfall.parquet", "loyo_oof_rainfall_cnn.parquet",
    "vic_daily.parquet",
    # v2 (Dong-spec) archive.  percell_v2.npz is the whole per-cell input in one
    # 140 MB file -- see preprocessing/pack_percell_kaggle.py for why the raw
    # 385 MB s2s/ directory is not shipped instead.
    "percell_v2_full.npz", "downscaling_v2_full.npz",
    # FutureTST (Ambika et al.) on the 19-monsoon record.  inflow_daily.parquet
    # stops at 2014 and forecast_catchment_met.parquet at lead 17, so without
    # these the kernel silently reruns the old 11-year, 308-init experiment.
    # raw_ec_catchment_prec.parquet is the catchment mean of ec_qm_v2_full.npz
    # -- 100 kB instead of shipping the 200 MB per-cell array.
    "inflow_daily_extended.parquet", "forecast_catchment_met_v2.parquet",
    # raw_ec_catchment_prec.parquet is a CACHE futuretst_prob.py rebuilds if
    # absent; the copy on this Mac was moved aside during the date-label repair.
    "loyo_oof_rainfall_v2.parquet",
    # Ensemble mean/spread/percentiles per (init, lead) for the probabilistic
    # run -- 742 kB, versus the 200 MB per-cell array it is derived from.
    "ec_ensemble_catchment.parquet",
]


def kaggle_cmd() -> list[str]:
    """How to invoke the Kaggle CLI.

    `pip install kaggle` puts the entry point next to the interpreter that ran
    pip, which on this Mac is the 3.14 framework Python -- a directory that is
    not on PATH.  Falling back to `-m kaggle` means the tool works without the
    user having to fix their PATH first.
    """
    onpath = shutil.which("kaggle")
    if onpath:
        return [onpath]
    sibling = Path(sys.executable).parent / "kaggle"
    if sibling.exists():
        return [str(sibling)]
    return [sys.executable, "-m", "kaggle"]


KAGGLE = None


def sh(cmd: list[str], check: bool = True) -> str:
    if cmd[0] == "kaggle":
        cmd = KAGGLE + cmd[1:]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.exit(f"$ {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.stdout + p.stderr


def require_setup() -> str:
    global KAGGLE
    KAGGLE = kaggle_cmd()
    probe = subprocess.run(KAGGLE + ["--version"], capture_output=True, text=True)
    if probe.returncode != 0 and "authentication" not in (probe.stdout + probe.stderr).lower():
        sys.exit(f"kaggle CLI not usable ({' '.join(KAGGLE)}).  Run:  pip install kaggle")
    # Two auth styles are in the wild.  The classic one is ~/.kaggle/kaggle.json
    # holding {"username", "key"}; kaggle >= 2.2 also accepts a bare access
    # token in ~/.kaggle/access_token or $KAGGLE_API_TOKEN, and resolves the
    # username from it server-side.  Support both, and read the username back
    # from the authenticated client rather than guessing.
    home = Path.home() / ".kaggle"
    if not (os.environ.get("KAGGLE_API_TOKEN")
            or (home / "access_token").exists() or (home / "kaggle.json").exists()):
        sys.exit("no Kaggle credentials -- save an API token to "
                 f"{home / 'access_token'} or {home / 'kaggle.json'}")
    if KAGGLE_USER:
        return KAGGLE_USER
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        user = api.config_values.get("username", "")
    except Exception as e:
        sys.exit(f"could not authenticate to Kaggle: {e}")
    if not user:
        sys.exit("authenticated, but no username resolved -- export KAGGLE_USERNAME")
    return user


def build_payload(stage: Path) -> None:
    """Stage the code and data that the kernel will see as its dataset."""
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "data" / "processed").mkdir(parents=True)

    for d in CODE_DIRS:
        src = ROOT / d
        if src.exists():
            shutil.copytree(src, stage / d,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "vic"))
    n = 0
    for f in DATA_FILES:
        src = ROOT / "data" / "processed" / f
        if src.exists():
            shutil.copy2(src, stage / "data" / "processed" / f)
            n += 1
    mb = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file()) / 1e6
    print(f"  payload: {n}/{len(DATA_FILES)} data files, {mb:.1f} MB total")


def push_dataset(stage: Path, user: str) -> str:
    ref = f"{user}/{DATASET_SLUG}"
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET_SLUG, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    print(f"  pushing dataset {ref} ...")
    # `datasets status` answers 403 for a dataset that does not exist and
    # "ready" once it does, so 403 is the create/update discriminator.
    # (`datasets list --mine` does NOT reliably list private datasets.)
    exists = "403" not in sh(["kaggle", "datasets", "status", ref], check=False)
    verb = ["version", "-m", "update"] if exists else ["create"]
    print(f"  {'new version of' if exists else 'creating'} {ref}")
    sh(["kaggle", "datasets"] + verb + ["-p", str(stage), "--dir-mode", "zip"])

    # Upload returns before Kaggle has finished unpacking.  Launching the kernel
    # now gives it an empty /kaggle/input and it dies on the first copytree, so
    # block until the dataset reports ready.
    print("  waiting for Kaggle to finish processing the upload ...", flush=True)
    for _ in range(60):
        st = sh(["kaggle", "datasets", "status", ref], check=False).strip().lower()
        if "ready" in st:
            # "ready" can reflect the PREVIOUS version while a new one is still
            # propagating -- which mounts an empty /kaggle/input and kills the
            # kernel on its first copytree.  Require it to stay ready.
            time.sleep(20)
            if "ready" in sh(["kaggle", "datasets", "status", ref], check=False).strip().lower():
                print("  dataset ready (confirmed)")
                return ref
        if "error" in st and "403" not in st:
            sys.exit(f"dataset failed to process: {st}")
        time.sleep(10)
    sys.exit("dataset still not ready after 10 min")


def push_kernel(stage: Path, user: str, dataset_ref: str, scripts: list[str],
                gpu: bool, slug: str = KERNEL_SLUG) -> str:
    ref = f"{user}/{slug}"
    kdir = stage.parent / "kernel"   # sibling inside .kaggle_build
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)

    # The kernel entry point: put the dataset on sys.path, chdir into it so every
    # C.PROCESSED path resolves, then exec the target script as __main__.
    (kdir / "run.py").write_text(f'''"""Auto-generated by tools/kaggle_run.py -- do not edit here."""
import os, runpy, shlex, shutil, sys
from pathlib import Path

# Do NOT assume the mount path.  A kernel created by CLI push can open in the
# editor without the dataset attached, and "Save & Run All" then runs with an
# empty /kaggle/input -- which used to fail here with a bare FileNotFoundError
# that said nothing about the cause.  Discover the mount, and if there is none,
# say exactly what IS there.
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
_want = INPUT / "{DATASET_SLUG}"
if _want.is_dir():
    SRC = _want
else:
    _dirs = sorted(d for d in INPUT.glob("*") if d.is_dir()) if INPUT.is_dir() else []
    print(f"expected {{_want}} -- not present.  /kaggle/input contains: "
          f"{{[d.name for d in _dirs]}}", flush=True)
    if not _dirs:
        raise SystemExit(
            "No dataset is mounted.  In the notebook editor: right sidebar -> "
            "Input -> + Add Input -> Your Datasets -> add '{DATASET_SLUG}', "
            "then Save & Run All.  The CLI metadata alone does not attach it.")
    # Take the first directory only as a LAST resort.  The dataset can mount a
    # level deeper (seen as /kaggle/input/datasets/<owner>/<slug>/), and copying
    # that wrapper gives a proj/ with no configs/ and no inflow/ -- the run then
    # dies with a FileNotFoundError on the script itself, which looks like a
    # missing file rather than a wrong mount.  Search for the real project root.
    def _find_root(base, depth=4):
        if (base / "configs" / "config.py").is_file():
            return base
        if depth:
            for d in sorted(x for x in base.glob("*") if x.is_dir()):
                hit = _find_root(d, depth - 1)
                if hit:
                    return hit
        return None
    SRC = next((r for r in (_find_root(d) for d in _dirs) if r), None) or _dirs[0]
    print(f"using {{SRC}} instead", flush=True)
    if not (SRC / "configs" / "config.py").is_file():
        raise SystemExit(
            f"{{SRC}} does not look like the project (no configs/config.py).  "
            "The dataset is mounted but its layout is unexpected; check the "
            "Input panel in the notebook editor.")
shutil.copytree(SRC, WORK / "proj", dirs_exist_ok=True)
os.chdir(WORK / "proj")
sys.path.insert(0, str(WORK / "proj"))

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
if torch.cuda.is_available():
    try:                                   # prove the device can run a kernel
        torch.nn.Conv2d(1, 1, 3).cuda()(torch.zeros(1, 1, 8, 8, device="cuda"))
        print("GPU smoke test: OK", flush=True)
    except Exception as exc:
        # DO NOT fall back to CPU on a GPU kernel.  Kaggle's CPU is slower than
        # the laptop that pushed the job -- a P100 run of futuretst_prob took
        # 2,393 s and 5,043 s for its first two folds against 690 s and 3,333 s
        # locally -- so the fallback quietly burns a 12-hour session to produce
        # two folds.  Failing in 12 seconds is strictly better: the fix is one
        # click in the UI, and it can only be applied if the run stops.
        raise SystemExit(
            f"GPU UNUSABLE ({{exc}}).  "
            "This is the P100/sm_60 case: Kaggle's PyTorch ships no kernels "
            "for it.  Refusing to continue on CPU -- Kaggle's CPU is slower "
            "than your Mac, so a CPU run here is worse than not running at "
            "all.  FIX: notebook -> Settings -> Accelerator -> GPU T4 x2 -> "
            "Save & Run All.  Confirm the first log line says Tesla T4.")

for _s in {scripts!r}:
    # A script may carry arguments: "cnn/loyo_percell_v2.py --ts-weight 66".
    # runpy does not parse a command line, so split it and plant sys.argv the
    # way the interpreter would.  Without this the args are silently dropped and
    # the script runs on its defaults -- which looks like a successful run.
    _parts = shlex.split(_s)
    _path, _args = _parts[0], _parts[1:]
    print(f"\\n{{'=' * 70}}\\nRUNNING {{_path}}  args={{_args}}\\n{{'=' * 70}}", flush=True)
    sys.argv = [_path] + _args
    runpy.run_path(_path, run_name="__main__")

# Surface results at the top level so `kaggle kernels output` returns them.
for sub in ("results", "checkpoints", "data/processed"):
    s = WORK / "proj" / sub
    if s.exists():
        shutil.copytree(s, WORK / sub.replace("/", "_"), dirs_exist_ok=True)
''')
    # Compile the generated kernel before it leaves this machine.  The template
    # is an f-string, so a stray \n or an unbalanced brace produces a file that
    # is only syntactically checked on Kaggle -- costing a push, a manual run,
    # and a "version failed" with the real cause buried in the run log.
    _src = (kdir / "run.py").read_text()
    try:
        compile(_src, "run.py", "exec")
    except SyntaxError as exc:
        bad = _src.splitlines()[max(0, (exc.lineno or 1) - 1)]
        sys.exit(f"generated kernel is not valid Python: {exc.msg} at line "
                 f"{exc.lineno}\n    {bad.strip()}\n"
                 "Fix the template in tools/kaggle_run.py -- nothing was pushed.")

    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": slug, "code_file": "run.py",
        "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": gpu, "enable_internet": False,
        "dataset_sources": [dataset_ref],
        "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    print(f"  pushing kernel {ref} (gpu={gpu}) ...")
    sh(["kaggle", "kernels", "push", "-p", str(kdir)])
    return ref


def wait(ref: str, poll: int = 60) -> str:
    print(f"  waiting on {ref} (polling every {poll}s; Ctrl-C is safe, the run continues)")
    t0 = time.time()
    while True:
        out = sh(["kaggle", "kernels", "status", ref], check=False)
        low = out.lower()
        mins = (time.time() - t0) / 60
        if "complete" in low:
            print(f"  complete after {mins:.0f} min")
            return "complete"
        if "error" in low or "cancel" in low:
            print(f"  FAILED after {mins:.0f} min:\n{out}")
            return "error"
        print(f"    [{mins:5.1f} min] {out.strip().splitlines()[-1][:90]}")
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script", nargs="+",
                    help="one or more scripts, run in order in a single session, "
                         "e.g. cnn/loyo_field.py inflow/futuretst_v2.py")
    ap.add_argument("--cpu", action="store_true", help="run without a GPU")
    ap.add_argument("--name", default=KERNEL_SLUG,
                    help="kernel slug; use a distinct one to run alongside another job")
    ap.add_argument("--no-wait", action="store_true", help="push and exit")
    ap.add_argument("--skip-data", action="store_true",
                    help="dataset is already current; push only the kernel.  NOTE: the "
                         "project CODE ships inside the dataset, so this also skips "
                         "code changes -- use it only when nothing under "
                         "configs/ cnn/ inflow/ preprocessing/ hydrology/ has changed")
    args = ap.parse_args()

    # A script entry may carry arguments, e.g.
    #     "cnn/loyo_percell_v2.py --ts-weight 66 --folds 6 --tag _b66"
    # Validate the PATH only; the kernel splits the same way (shlex) and plants
    # sys.argv before runpy, so the arguments reach the script.
    import shlex as _shlex
    for s in args.script:
        _p = _shlex.split(s)[0] if s.strip() else s
        if not (ROOT / _p).exists():
            sys.exit(f"no such script: {_p}")
    user = require_setup()

    build = ROOT / ".kaggle_build"
    stage = build / "payload"
    print("1/4  bundling code + data")
    build_payload(stage)
    if args.skip_data:
        ds = f"{user}/{DATASET_SLUG}"
        print(f"2/4  dataset: skipped, reusing {ds}")
        # The code lives in the dataset, so --skip-data means Kaggle runs
        # whatever code was uploaded last time.  Warn loudly: a stale run that
        # looks successful is worse than a failed one.
        # Was max() over a generator OF generators, which raised TypeError and
        # made --skip-data unusable.  Flatten first, then compare mtimes, and
        # name the most recently edited file so the warning is checkable.
        pys = [f for d in CODE_DIRS if (ROOT / d).exists()
               for f in (ROOT / d).rglob("*.py")]
        print("     !! code is NOT re-uploaded with --skip-data; the kernel will run "
              "the previously uploaded version")
        if pys:
            newest = max(pys, key=lambda f: f.stat().st_mtime)
            import datetime as _dt
            when = _dt.datetime.fromtimestamp(newest.stat().st_mtime)
            print(f"     most recent local edit: {newest.relative_to(ROOT)} "
                  f"({when:%Y-%m-%d %H:%M}) -- if that matters, drop --skip-data")
    else:
        print("2/4  dataset")
        ds = push_dataset(stage, user)
    print("3/4  kernel")
    kn = push_kernel(stage, user, ds, args.script, gpu=not args.cpu, slug=args.name)
    if args.no_wait:
        shutil.rmtree(build, ignore_errors=True)
        print(f"\npushed.  https://www.kaggle.com/code/{kn}")
        if not args.cpu:
            print("\n  !! MANUAL STEP REQUIRED to get a usable GPU:")
            print("     Kaggle assigns a Tesla P100 (sm_60) by default, and Kaggle's own")
            print("     PyTorch build ships no kernels for it -- every conv fails.  The")
            print("     `--accelerator` CLI flag does NOT override this, and a CLI push")
            print("     RESETS whatever the web UI had set.  So, every push:")
            print("       1. open the notebook link above")
            print("       2. Settings -> Accelerator -> GPU T4 x2")
            print("       3. Save & Run All")
        return
    print("4/4  running")
    shutil.rmtree(build, ignore_errors=True)     # staging is disposable
    if wait(kn) != "complete":
        sys.exit(f"see https://www.kaggle.com/code/{kn}")

    dst = ROOT / "kaggle_output"
    dst.mkdir(exist_ok=True)
    sh(["kaggle", "kernels", "output", kn, "-p", str(dst)])
    print(f"\ndownloaded -> {dst}")
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dst)}  ({p.stat().st_size/1e3:.0f} KB)")
    print("\nCopy what you want into results/ or data/processed/ -- "
          "nothing is overwritten automatically.")


if __name__ == "__main__":
    main()
