"""Ablation study: FID vs. guidance scale and FID vs. sampling steps.

Writes a Markdown table to results.md.
  python ablation.py --ckpt checkpoints/ema.pt --n 5000
"""

import argparse

import torch

from fid import compute_fid
from model import DiT

CFG_SCALES = [1.0, 2.0, 4.0]
STEP_COUNTS = [10, 50, 250]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ema.pt")
    ap.add_argument("--n", type=int, default=5000, help="images per FID estimate")
    ap.add_argument("--out", default="results.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiT().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    lines = ["## Ablation results", "",
             f"FID computed over {args.n} generated images vs. the "
             "FashionMNIST test set.", ""]

    # Guidance scale (sampling steps held at 50).
    lines += ["### Classifier-free guidance scale (steps = 50)", "",
              "| cfg scale | FID |", "|---|---|"]
    for cfg in CFG_SCALES:
        fid = compute_fid(model, device, n=args.n, cfg_scale=cfg, num_steps=50)
        print(f"cfg={cfg}: FID={fid:.2f}")
        lines.append(f"| {cfg} | {fid:.2f} |")

    # Sampling steps (guidance held at 2.0).
    lines += ["", "### Sampling steps (cfg = 2.0)", "",
              "| Euler steps | FID |", "|---|---|"]
    for steps in STEP_COUNTS:
        fid = compute_fid(model, device, n=args.n, cfg_scale=2.0, num_steps=steps)
        print(f"steps={steps}: FID={fid:.2f}")
        lines.append(f"| {steps} | {fid:.2f} |")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
