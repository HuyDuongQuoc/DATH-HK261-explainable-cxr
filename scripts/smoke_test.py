import argparse
import json
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        required=True
    )

    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "smoke_test_checkpoint.txt"
    )

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        f.write("Checkpoint test successful.\n")
        f.write(f"Created at: {datetime.now()}\n")

    result = {
        "experiment": "smoke_test",
        "model": "dummy",
        "auroc": 0.85,
        "f1": 0.80,
        "status": "success"
    }

    result_path = os.path.join(
        args.result_dir,
        "metrics.json"
    )

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("Smoke test completed.")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()