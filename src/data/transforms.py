from torchvision import transforms


def build_image_transform(mean, std):
    return transforms.Compose([
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])