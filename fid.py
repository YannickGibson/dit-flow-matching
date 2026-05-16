"""FID evaluation for a trained DiT checkpoint (FashionMNIST or CIFAR-10).

`compute_fid` is reusable from other scripts (see ablation.py).
  python fid.py --dataset cifar10 --n 10000 --cfg 2.0 --steps 50
"""

import argparse
import os
import shutil

import torch
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image

from data_utils import DATASET_CONFIG, build_dataset
from flow import sample
from model import DiT


def dump_real_images(dataset, limit=10000):
    """Save the test split as PNGs once (in [0, 1]), as the FID reference."""
    out_dir = f"fid_real_{dataset}"
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= limit:
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    data = build_dataset(dataset, train=False, normalize=False)
    for i in range(min(limit, len(data))):
        img, _ = data[i]
        save_image(img, os.path.join(out_dir, f"{i:05d}.png"))
    return out_dir


def generate_fake_images(model, out_dir, n, cfg_scale, num_steps, device, batch=250):
    """Sample `n` images from the model into a fresh folder."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    done = 0
    while done < n:
        b = min(batch, n - done)
        labels = torch.randint(0, 10, (b,), device=device)
        imgs = sample(model, b, labels, num_steps=num_steps,
                      cfg_scale=cfg_scale, device=device) * 0.5 + 0.5
        for j in range(b):
            save_image(imgs[j], os.path.join(out_dir, f"{done + j:05d}.png"))
        done += b
    return out_dir


def compute_fid(model, dataset, device, n=10000, cfg_scale=2.0, num_steps=50):
    """Return FID between `n` generated images and the dataset's test split."""
    real_dir = dump_real_images(dataset, limit=n)
    fake_dir = generate_fake_images(model, f"fid_fake_{dataset}", n,
                                    cfg_scale, num_steps, device)
    metrics = calculate_metrics(input1=fake_dir, input2=real_dir,
                                fid=True, verbose=False)
    return metrics["frechet_inception_distance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fashionmnist", "cifar10"],
                    default="fashionmnist")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    ckpt = args.ckpt or f"checkpoints/{args.dataset}_ema.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiT(**DATASET_CONFIG[args.dataset]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    fid = compute_fid(model, args.dataset, device, n=args.n,
                      cfg_scale=args.cfg, num_steps=args.steps)
    print(f"FID [{args.dataset}] (n={args.n}, cfg={args.cfg}, "
          f"steps={args.steps}): {fid:.2f}")


if __name__ == "__main__":
    main()
