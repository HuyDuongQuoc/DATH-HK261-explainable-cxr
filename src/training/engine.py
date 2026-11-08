import time

import numpy as np
import torch
from tqdm.auto import tqdm

from src.losses.localization_loss import localization_loss
from src.models.resnet import forward_with_activation

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
    collect_predictions=True,
    model_name=None,
    loc_weight=0.0,
):
    training = optimizer is not None
    use_loc = loc_weight > 0
    if use_loc and model_name is None:
        raise ValueError("model_name is required when loc_weight > 0")
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

    loss_total_tensor = torch.zeros((), dtype=torch.float64, device=device)
    cls_total_tensor = torch.zeros((), dtype=torch.float64, device=device)
    loc_total_tensor = torch.zeros((), dtype=torch.float64, device=device)

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
                if use_loc:
                    logits, activation = forward_with_activation(
                        model,
                        model_name,
                        images,
                    )
                else:
                    logits = model(images)

                if logits.shape != labels.shape:
                    raise ValueError(
                        "Model output and label "
                        "shapes do not match"
                    )

                cls_loss = criterion(
                    logits,
                    labels,
                )

            if use_loc:
                # L_loc tính ở fp32 ngoài autocast để tỉ lệ năng lượng không mất độ chính xác
                loc_loss = localization_loss(
                    activation.float(),
                    batch["union_masks"].to(device, non_blocking=device.type == "cuda"),
                    batch["mask_valid"].to(device, non_blocking=device.type == "cuda"),
                )
                loss = cls_loss.float() + loc_weight * loc_loss
            else:
                loc_loss = torch.zeros_like(cls_loss)
                loss = cls_loss

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

            loss_total_tensor += loss.detach().to(torch.float64) * batch_size
            cls_total_tensor += cls_loss.detach().to(torch.float64) * batch_size
            loc_total_tensor += loc_loss.detach().to(torch.float64) * batch_size

            image_count += batch_size

            if collect_predictions:
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
                "img/s": f"{image_count / elapsed:.1f}",
            }
            if show_progress:
                total_loss = float(loss_total_tensor.item())
                postfix["loss"] = f"{total_loss / image_count:.4f}"
            if device.type == "cuda":
                postfix["gpu_GB"] = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f}"
            progress.set_postfix(postfix, refresh=False)

    if image_count == 0:
        raise ValueError("Empty DataLoader")

    elapsed_seconds = time.perf_counter() - started_at
    total_loss = float(loss_total_tensor.item())
    peak_gpu_memory_allocated_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )
    peak_gpu_memory_reserved_gb = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )

    result = {
        "loss": total_loss / image_count,
        "cls_loss": float(cls_total_tensor.item()) / image_count,
        "loc_loss": float(loc_total_tensor.item()) / image_count,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": image_count / elapsed_seconds,
        "peak_gpu_memory_gb": peak_gpu_memory_allocated_gb,
        "peak_gpu_memory_allocated_gb": peak_gpu_memory_allocated_gb,
        "peak_gpu_memory_reserved_gb": peak_gpu_memory_reserved_gb,
    }
    if collect_predictions:
        result.update({
            "targets": np.concatenate(targets, axis=0),
            "probabilities": np.concatenate(probabilities, axis=0),
            "image_ids": image_ids,
        })
    return result
