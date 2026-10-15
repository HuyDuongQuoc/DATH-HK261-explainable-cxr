import numpy as np
import torch

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    image_count = 0

    targets = []
    probabilities = []
    image_ids = []

    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device)

            labels = batch["labels"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

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

    if image_count == 0:
        raise ValueError("Empty DataLoader")

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
    }