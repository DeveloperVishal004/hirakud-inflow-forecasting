"""Make RVIC 1.1.0 importable on modern Python.

RVIC's last release is from 2017 and it is effectively unmaintained, so it
still uses APIs that Python has since removed.  These are renames, not
behaviour changes, and patching them is safe:

  SafeConfigParser -> ConfigParser   Python 3.2 renamed it and deprecated the
                                     old alias; 3.12 removed the alias.

Patching site-packages is normally the wrong instinct -- it is invisible and
lost on reinstall -- so this script exists to make the change explicit,
idempotent and re-runnable rather than something done once by hand and
forgotten.  Re-run it after any `pip install rvic`.

Run:  python hydrology/vic/patch_rvic.py
"""

import sys
from pathlib import Path

PATCHES = [
    ("core/pycompat.py",
     "from configparser import SafeConfigParser",
     "from configparser import ConfigParser as SafeConfigParser"),
    # sysconfig's 'SO' key was deprecated in 3.4 and removed in 3.8; the
    # replacement 'EXT_SUFFIX' carries the same value (e.g. '.cpython-314-
    # darwin.so').  Unpatched, get_config_var returns None and the module
    # fails at import with a TypeError on string concatenation.
    ("core/convolution_wrapper.py",
     "sysconfig.get_config_var('SO')",
     "sysconfig.get_config_var('EXT_SUFFIX')"),
    # RVIC's own bug, not a Python change: isint("1.0") is True, because it
    # tests float(x) == int(float(x)) -- but int("1.0") then raises ValueError.
    # Any whole-number float in the config crashes the parser, which matters
    # because VELOCITY is a routing parameter people calibrate: 1.5 works, 2.0
    # does not.
    ("core/config.py",
     "            return int(value)",
     "            return int(float(value))"),
    # DataFrame.ix was deprecated in pandas 0.20 and removed in 1.0.  For
    # scalar label-based assignment .loc is the direct replacement.
    ("parameters.py",
     "pour_points.ix[i, 'names']",
     "pour_points.loc[i, 'names']"),
    # np.float was an alias for the builtin float, deprecated in NumPy 1.20 and
    # removed in 1.24.  np.finfo(float) is the identical call.
    ("parameters.py",
     "np.finfo(np.float).resolution",
     "np.finfo(float).resolution"),
    # Python 3 made `/` true division, so this yields a float where Python 2
    # gave an int -- and it is then used as an array dimension.  Integer
    # division is what was meant.
    # SECSPERDAY is itself a float, so `//` still yields a float; the value is
    # used as an array dimension and must be a genuine int.
    ("core/param_file.py",
     "        subset_length = (options['SUBSET_DAYS'] *\n"
     "                         SECSPERDAY // routing['OUTPUT_INTERVAL'])",
     "        subset_length = int(options['SUBSET_DAYS'] *\n"
     "                            SECSPERDAY // routing['OUTPUT_INTERVAL'])"),
    # Same true-division problem again, here producing float slice bounds for
    # the unit-hydrograph window.
    ("core/param_file.py",
     "        d_left = -1 * subset_length / 2\n"
     "        d_right = subset_length / 2",
     "        d_left = -1 * subset_length // 2\n"
     "        d_right = subset_length // 2"),
    # Under Python 3 the outlet name arrives as numpy.bytes_, but netCDF4's
    # stringtochar calls .encode() on it.  Decoding to str first is what the
    # Python 2 path effectively did.
    ("core/write.py",
     "    char_names = stringtochar(outlet_name)",
     "    char_names = stringtochar(np.array(\n"
     "        [n.decode() if isinstance(n, bytes) else str(n)\n"
     "         for n in np.atleast_1d(outlet_name)], dtype='U'))"),
]


def main() -> None:
    try:
        import rvic
    except ImportError:
        sys.exit("rvic is not installed:  pip install rvic")

    root = Path(rvic.__file__).parent
    print(f"rvic at {root}")

    changed = 0
    for rel, old, new in PATCHES:
        f = root / rel
        if not f.exists():
            print(f"  {rel:22s} MISSING -- rvic layout differs from expected")
            continue
        text = f.read_text()
        if new in text:
            print(f"  {rel:22s} already patched")
        elif old in text:
            f.write_text(text.replace(old, new))
            print(f"  {rel:22s} patched")
            changed += 1
        else:
            print(f"  {rel:22s} pattern absent -- upstream may have fixed it")

    # Import is the real test; the patch list above is necessary, not
    # provably sufficient, for whatever Python version is in use.
    print()
    try:
        import importlib
        importlib.invalidate_caches()
        for mod in ("rvic.parameters", "rvic.convolution"):
            importlib.import_module(mod)
            print(f"  import {mod:20s} OK")
    except Exception as exc:  # noqa: BLE001
        print(f"  import FAILED: {type(exc).__name__}: {exc}")
        print("\nFAIL -- another incompatibility remains; add it to PATCHES.")
        sys.exit(1)

    print(f"\nPASS ({changed} file(s) changed this run)")


if __name__ == "__main__":
    main()
