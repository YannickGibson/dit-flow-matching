"""Dataset configs, transforms, and loaders for FashionMNIST and CIFAR-10.

Both are 32x32 with 10 classes. FashionMNIST is 28x28 grayscale (padded to 32);
CIFAR-10 is already 32x32 RGB. The DiT is scaled up for the harder RGB task.
"""

from torchvision import datasets, transforms

# DiT constructor kwargs per dataset (splatted into DiT(**...)).
DATASET_CONFIG = {
    "fashionmnist": dict(in_ch=1, num_classes=10, hidden=256, depth=8, heads=4),
    "cifar10": dict(in_ch=3, num_classes=10, hidden=384, depth=12, heads=6),
}

CLASS_NAMES = {
    "fashionmnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"],
}


def build_transform(dataset, train, normalize=True):
    """Compose transforms. normalize=False yields [0, 1] images (for FID refs)."""
    ops = []
    if dataset == "fashionmnist":
        ops.append(transforms.Pad(2))  # 28x28 -> 32x32
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    if normalize:
        ch = DATASET_CONFIG[dataset]["in_ch"]
        ops.append(transforms.Normalize((0.5,) * ch, (0.5,) * ch))  # -> [-1, 1]
    return transforms.Compose(ops)


def build_dataset(dataset, train=True, normalize=True):
    tf = build_transform(dataset, train, normalize)
    cls = datasets.FashionMNIST if dataset == "fashionmnist" else datasets.CIFAR10
    return cls("data", train=train, download=True, transform=tf)
