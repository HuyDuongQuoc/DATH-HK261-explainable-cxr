import time

import numpy as np
import torch
from tqdm.auto import tqdm

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp=False,
    show_progress=False,
    description=None,
):
    training = optimizer is not None
    amp_enabled = bool(use_amp and device.type == "cuda")
    if training and amp_enabled and scaler is None:
        raise ValueError("AMP training requires a GradScaler")

    model.train(training)

    total_loss = 0.0
    image_count = 0

    targets = []
    probabilities = []
    image_ids = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started_at = time.perf_counter()
    progress = tqdm(
        loader,
        desc=description,
        disable=not show_progress,
        unit="batch",
        leave=True,
    )

    with torch.set_grad_enabled(training):
        for batch in progress:
            images = batch["image"].to(
                device,
                non_blocking=device.type == "cuda",
            )

            labels = batch["labels"].to(
                device,
                non_blocking=device.type == "cuda",
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)

                if logits.shape != labels.shape:
                    raise ValueError(
                        "Model output and label "
                        "shapes do not match"
                    )

                loss = criterion(
                    logits,
                    labels,
                )

            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")

            if training:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = len(images)

            total_loss += loss.item() * batch_size

            image_count += batch_size

            targets.append(
                labels.detach()
                .cpu()
                .numpy()
            )

            probabilities.append(
                logits.detach()
                .sigmoid()
                .cpu()
                .numpy()
            )

            image_ids.extend(
                batch["image_id"]
            )

            elapsed = max(time.perf_counter() - started_at, 1e-9)
            postfix = {
                "loss": f"{total_loss / image_count:.4f}",
                "img/s": f"{image_count / elapsed:.1f}",
            }
            if device.type == "cuda":
                postfix["gpu_GB"] = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f}"
            progress.set_postfix(postfix, refresh=False)

    if image_count == 0:
        raise ValueError("Empty DataLoader")

    elapsed_seconds = time.perf_counter() - started_at
    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )

    return {
        "loss": total_loss / image_count,
        "targets": np.concatenate(
            targets,
            axis=0,
        ),
        "probabilities": np.concatenate(
            probabilities,
            axis=0,
        ),
        "image_ids": image_ids,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": image_count / elapsed_seconds,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    }
