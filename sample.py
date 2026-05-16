"""Generate a sample grid from a trained EMA checkpoint.

Example:
  python sample.py --ckpt checkpoints/ema.pt --cfg 2.0 --steps 50
"""

import argparse

import torch
from torchvision.utils import save_image

from flow import sample
from model import DiT

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ema.pt")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="grid.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiT().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # 10 samples per class, one class per row.
    labels = torch.arange(10, device=device).repeat_interleave(10)
    imgs = sample(model, 100, labels, num_steps=args.steps,
                  cfg_scale=args.cfg, device=device)
    save_image(imgs * 0.5 + 0.5, args.out, nrow=10)
    print(f"saved {args.out} (cfg={args.cfg}, steps={args.steps})")


if __name__ == "__main__":
    main()
