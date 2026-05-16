"""Train the DiT on FashionMNIST with flow matching.

Single-A100 friendly. Run:  python train.py
Optional experiment tracking:  python train.py --wandb
"""

import argparse
import copy
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from flow import flow_matching_loss, sample
from model import DiT


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N steps (0 = no limit); use for smoke tests")
    ap.add_argument("--wandb", action="store_true",
                    help="log metrics and sample grids to Weights & Biases")
    ap.add_argument("--wandb-project", type=str, default="dit-fashionmnist")
    args = ap.parse_args()

    # Optional experiment tracking. Kept fully optional so the repo runs
    # without a W&B account; `wandb` is only imported when --wandb is passed.
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # FashionMNIST: 28x28 grayscale -> pad to 32x32, scale to [-1, 1].
    tf = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    data = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)

    model = DiT().to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiT params: {n_params/1e6:.1f}M | device: {device}")
    if run is not None:
        run.summary["params_millions"] = n_params / 1e6

    step = 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device, enabled=use_amp):
                loss = flow_matching_loss(model, x, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model, args.ema_decay)
            step += 1
            if run is not None and step % 50 == 0:
                run.log({"loss": loss.item(), "lr": args.lr, "epoch": epoch},
                        step=step)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                break

        # Preview grid: one row per class, sampled from the EMA weights.
        labels = torch.arange(10, device=device).repeat_interleave(10)
        imgs = sample(ema, 100, labels, num_steps=50, cfg_scale=2.0, device=device)
        grid_path = f"samples/epoch_{epoch:03d}.png"
        save_image(imgs * 0.5 + 0.5, grid_path, nrow=10)
        torch.save(ema.state_dict(), os.path.join(args.out, "ema.pt"))
        if run is not None:
            import wandb
            run.log({"samples": wandb.Image(grid_path)}, step=step)

        if args.max_steps and step >= args.max_steps:
            print(f"reached max-steps={args.max_steps}, stopping")
            break

    print("done. EMA checkpoint:", os.path.join(args.out, "ema.pt"))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
