"""
Kernel Kaggle (GPU): tìm bundle trong /kaggle/input rồi chạy scripts/kaggle_eval.py.
Bundle = Kaggle dataset tạo bởi scripts/make_eval_bundle.py. Kết quả ở /kaggle/working/runs/.

Khi upload bằng `kaggle datasets create --dir-mode zip`, mỗi thư mục con có thể
nằm dưới dạng <tên>.zip nếu Kaggle không tự giải nén; kernel tự giải nén vào /kaggle/tmp.
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
        raise FileNotFoundError(f"Thiếu {part} trong {source}: {sorted(p.name for p in source.iterdir())}")
    extract(archives[0], bundle / part)
    nested = bundle / part / part  # zip có thể chứa thư mục gốc cùng tên
    if nested.is_dir() and len(list((bundle / part).iterdir())) == 1:
        nested.rename(bundle / f"{part}_tmp")
        (bundle / part).rmdir()
        (bundle / f"{part}_tmp").rename(bundle / part)


def find_bundle():
    """
    Ghép bundle từ hai dataset: dữ liệu (bundle_manifest.json, data/, priors/, runs/)
    và code (code/scripts/kaggle_eval.py hoặc code.zip). Có thể là một dataset chung.
    """
    manifests = list(INPUT.rglob("bundle_manifest.json"))
    if not manifests:
        raise FileNotFoundError("Không tìm thấy bundle_manifest.json trong /kaggle/input")
    data_source = manifests[0].parent
    bundle = Path("/kaggle/tmp/bundle")
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "bundle_manifest.json").write_bytes(manifests[0].read_bytes())
    for part in ("data", "priors", "runs"):
        _materialize(data_source, part, bundle)

    # Thư mục code = thư mục chứa scripts/kaggle_eval.py (Kaggle có thể bỏ hoặc giữ
    # thư mục gốc "code/" khi giải nén code.zip, và mount dưới /kaggle/input/datasets/<user>/).
    code_hits = sorted(INPUT.rglob("scripts/kaggle_eval.py"), key=lambda p: len(p.parts))
    if code_hits:
        if not (bundle / "code").exists():
            os.symlink(code_hits[0].parents[1], bundle / "code")
    else:
        archives = list(INPUT.rglob("code.zip")) + list(INPUT.rglob("code.tar*"))
        if not archives:
            raise FileNotFoundError("Không tìm thấy scripts/kaggle_eval.py hay code.zip trong /kaggle/input")
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
