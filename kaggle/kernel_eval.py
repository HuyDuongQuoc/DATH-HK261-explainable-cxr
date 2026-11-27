"""
Kaggle GPU kernel: find the bundle under /kaggle/input and run scripts/kaggle_eval.py.
The bundle is a Kaggle dataset built by scripts/make_eval_bundle.py. Results go to
/kaggle/working/runs/.

With `kaggle datasets create --dir-mode zip`, each subfolder may arrive as <name>.zip
if Kaggle does not unpack it; the kernel then unpacks it into /kaggle/tmp.
"""
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target)


def _materialize(source, part, bundle):
    if (bundle / part).exists():
        return
    if (source / part).is_dir():
        os.symlink(source / part, bundle / part)
        return
    archives = [p for p in source.iterdir() if p.name.startswith(part + ".")]
    if not archives:
        raise FileNotFoundError(f"Missing {part} in {source}: {sorted(p.name for p in source.iterdir())}")
    extract(archives[0], bundle / part)
    nested = bundle / part / part  # the zip may contain a top-level folder with the same name
    if nested.is_dir() and len(list((bundle / part).iterdir())) == 1:
        nested.rename(bundle / f"{part}_tmp")
        (bundle / part).rmdir()
        (bundle / f"{part}_tmp").rename(bundle / part)


def find_bundle():
    """
    Assemble the bundle from two datasets: data (bundle_manifest.json, data/, priors/,
    runs/) and code (code/scripts/kaggle_eval.py or code.zip). Both may be one dataset.
    """
    manifests = list(INPUT.rglob("bundle_manifest.json"))
    if not manifests:
        raise FileNotFoundError("bundle_manifest.json not found under /kaggle/input")
    data_source = manifests[0].parent
    bundle = Path("/kaggle/tmp/bundle")
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "bundle_manifest.json").write_bytes(manifests[0].read_bytes())
    for part in ("data", "priors", "runs"):
        _materialize(data_source, part, bundle)

    # The code folder is the one that contains scripts/kaggle_eval.py (Kaggle may drop or
    # keep the top-level "code/" folder when unpacking, and mounts under /kaggle/input/datasets/<user>/).
    code_hits = sorted(INPUT.rglob("scripts/kaggle_eval.py"), key=lambda p: len(p.parts))
    if code_hits:
        if not (bundle / "code").exists():
            os.symlink(code_hits[0].parents[1], bundle / "code")
    else:
        archives = list(INPUT.rglob("code.zip")) + list(INPUT.rglob("code.tar*"))
        if not archives:
            raise FileNotFoundError("Neither scripts/kaggle_eval.py nor code.zip found under /kaggle/input")
        _materialize(archives[0].parent, "code", bundle)
    print("code dir:", (bundle / "code").resolve(), flush=True)
    return bundle


bundle = find_bundle()
print("bundle:", bundle, sorted(p.name for p in bundle.iterdir()), flush=True)
subprocess.run(["nvidia-smi"], check=False)
subprocess.run(
    [sys.executable, str(bundle / "code" / "scripts" / "kaggle_eval.py"),
     "--bundle", str(bundle), "--output", str(WORK)],
    check=True,
)
