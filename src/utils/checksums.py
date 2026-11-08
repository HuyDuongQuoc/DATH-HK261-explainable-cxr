import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_file_hashes(protocol_dir):
    protocol_dir = Path(protocol_dir)
    hashes = {}
    for path in sorted(protocol_dir.iterdir()):
        if not path.is_file() or path.name == "checksums.json":
            continue
        if path.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}:
            continue
        hashes[path.name] = file_sha256(path)
    return hashes


def write_protocol_checksums(protocol_dir):
    protocol_dir = Path(protocol_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    output = protocol_dir / "checksums.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(protocol_file_hashes(protocol_dir), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
