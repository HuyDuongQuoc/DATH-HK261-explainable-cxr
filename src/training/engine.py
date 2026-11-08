import numpy as np
import torch

from src.losses.localization_loss import localization_loss
from src.models.resnet import forward_with_activation


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    model_name=None,
    loc_weight=0.0,
):
    training = optimizer is not None

    model.train(training)

    use_loc = loc_weight > 0

    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0
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
                loc_loss = localization_loss(
                    activation,
                    batch["union_masks"].to(device),
                    batch["mask_valid"].to(device),
                )
                loss = cls_loss + loc_weight * loc_loss
            else:
                loc_loss = torch.zeros_like(cls_loss)
                loss = cls_loss

            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")

            if training:
                loss.backward()
                optimizer.step()

            batch_size = len(images)

            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            total_loc_loss += loc_loss.item() * batch_size

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
        "cls_loss": total_cls_loss / image_count,
        "loc_loss": total_loc_loss / image_count,
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