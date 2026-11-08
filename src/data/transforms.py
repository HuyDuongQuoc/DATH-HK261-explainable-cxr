from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_image_transform(mean, std, augmentation=None):
    steps = [
        transforms.Grayscale(
            num_output_channels=3
        ),
    ]

    if augmentation is not None:
        steps.extend([
            transforms.RandomAffine(
                degrees=float(augmentation["rotation_degrees"]),
                translate=(
                    float(augmentation["translate_fraction"]),
                    float(augmentation["translate_fraction"]),
                ),
                scale=tuple(float(value) for value in augmentation["scale"]),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=float(augmentation["brightness"]),
                        contrast=float(augmentation["contrast"]),
                    )
                ],
                p=float(augmentation["intensity_probability"]),
            ),
        ])

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    return transforms.Compose(steps)
