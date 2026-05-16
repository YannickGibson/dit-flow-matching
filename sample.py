"""Generate a labeled sample grid from a trained EMA checkpoint.

  python sample.py --dataset fashionmnist
  python sample.py --dataset cifar10 --cfg 2.0 --steps 50
"""

import argparse

import torch

from data_utils import CLASS_NAMES, DATASET_CONFIG
from flow import sample
from grid import save_labeled_grid
from model import DiT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fashionmnist", "cifar10"],
                    default="fashionmnist")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path (default: checkpoints/<dataset>_ema.pt)")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="grid.png")
    args = ap.parse_args()

    ckpt = args.ckpt or f"checkpoints/{args.dataset}_ema.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiT(**DATASET_CONFIG[args.dataset]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # 10 samples per class, one class per row.
    labels = torch.arange(10, device=device).repeat_interleave(10)
    imgs = sample(model, 100, labels, num_steps=args.steps,
                  cfg_scale=args.cfg, device=device)
    save_labeled_grid(imgs * 0.5 + 0.5, CLASS_NAMES[args.dataset], args.out)
    print(f"saved {args.out}  ({args.dataset}, cfg={args.cfg}, steps={args.steps})")


if __name__ == "__main__":
    main()
