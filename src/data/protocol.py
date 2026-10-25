import json
from pathlib import Path


def load_split_metadata(split_root):
    metadata_path = Path(split_root) / "split_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assignment = metadata.get("assignment_sha256")
    if not isinstance(assignment, str) or not assignment:
        raise ValueError("Split metadata is missing assignment_sha256")
    return metadata


def validate_checkpoint_protocol(
    checkpoint,
    model_name,
    findings,
    assignment_sha256,
):
    if checkpoint.get("model_name") != model_name:
        raise ValueError("Checkpoint backbone mismatch")
    if checkpoint.get("findings") != list(findings):
        raise ValueError("Checkpoint class order mismatch")
    if checkpoint.get("split_assignment_sha256") != assignment_sha256:
        raise ValueError("Checkpoint split fingerprint mismatch")
