"""FID evaluation for the DiT FashionMNIST model.

`compute_fid` is reusable from other scripts (see ablation.py).
Standalone:
  python fid.py --ckpt checkpoints/ema.pt --n 10000 --cfg 2.0 --steps 50
"""

import argparse
import os
import shutil

import torch
from torch_fidelity import calculate_metrics
from torchvision import datasets, transforms
from torchvision.utils import save_image

from flow import sample
from model import DiT


def dump_real_images(out_dir="fid_real", limit=10000):
    """Save FashionMNIST test images (padded to 32x32) once, for reuse."""
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= limit:
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    data = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
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


def compute_fid(model, device, n=10000, cfg_scale=2.0, num_steps=50,
                real_dir="fid_real", fake_dir="fid_fake"):
    """Return FID between `n` generated images and the FashionMNIST test set."""
    dump_real_images(real_dir, limit=n)
    generate_fake_images(model, fake_dir, n, cfg_scale, num_steps, device)
    metrics = calculate_metrics(input1=fake_dir, input2=real_dir,
                                fid=True, verbose=False)
    return metrics["frechet_inception_distance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ema.pt")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiT().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    fid = compute_fid(model, device, n=args.n,
                      cfg_scale=args.cfg, num_steps=args.steps)
    print(f"FID (n={args.n}, cfg={args.cfg}, steps={args.steps}): {fid:.2f}")


if __name__ == "__main__":
    main()
