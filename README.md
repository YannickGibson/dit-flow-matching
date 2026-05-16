# DiT + Flow Matching on FashionMNIST

Class-conditional image generation with a **Diffusion Transformer**, trained
from scratch on FashionMNIST.

A from-scratch reproduction of the DiT architecture
([Peebles & Xie, 2022](https://arxiv.org/abs/2212.09748)), trained with a
**conditional flow-matching** objective
([Lipman et al., 2023](https://arxiv.org/abs/2210.02747)) instead of the
original DDPM noise prediction. Pixel-space 32×32 grayscale - no VAE, runs on a
single GPU.

![Generated samples](assets/samples.png)

*Class-conditional samples after 120 epochs - one row per class
(T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag,
Ankle boot), classifier-free guidance scale 2.0, 50 sampling steps.*

## Quickstart

```bash
git clone https://github.com/YannickGibson/dit-fashionmnist.git && cd dit-fashionmnist
uv sync                       # creates .venv and installs everything

uv run python train.py        # train (~2-4h on one A100)
uv run python train.py --wandb   # ...with live Weights & Biases tracking
uv run python sample.py       # generate a sample grid -> grid.png
uv run python fid.py          # FID against the FashionMNIST test set
uv run python ablation.py     # ablation tables -> results.md
```

No `uv`? Fall back to `pip install -r requirements.txt` then drop the
`uv run` prefix.

## What's in here

| File | Purpose |
|---|---|
| `model.py` | DiT - patchify, transformer blocks with **adaLN-zero**, linear head |
| `flow.py` | Flow-matching loss + Euler ODE sampler with classifier-free guidance |
| `train.py` | Training loop (mixed precision, EMA); writes sample grids per epoch |
| `sample.py` | Generate a 10×10 class-conditional grid from a checkpoint |
| `fid.py` | FID evaluation; `compute_fid()` is reusable |
| `ablation.py` | FID vs. guidance scale and vs. sampling steps → `results.md` |

The `slurm/` folder holds example SLURM batch scripts - adapt the `#SBATCH`
directives to your scheduler, or ignore them and run the `uv run` commands
directly.

## How it works

**DiT.** The image is split into 2×2 patches (→ 256 tokens) and processed by a
transformer. Timestep and class are summed into a conditioning vector `c`;
each block derives shift/scale/gate parameters from `c` (**adaLN-zero**). The
gate projection is zero-initialized, so every block starts as an identity map
and the network eases into using conditioning during training.

**Flow matching.** Along the linear path `x_t = (1-t)·noise + t·data`, the
ideal velocity is the constant `data - noise`. The network regresses it with
plain MSE. Sampling integrates `dx/dt = v(x, t)` from noise (`t=0`) to data
(`t=1`) - far fewer steps than DDPM.

**Classifier-free guidance.** The class label is dropped 10% of the time in
training; at sampling, the conditional and unconditional velocities are
extrapolated to sharpen class identity.

## Experiment tracking

Pass `--wandb` to `train.py` to log loss, learning rate, and per-epoch sample
grids to [Weights & Biases](https://wandb.ai). Without the flag, training runs
exactly as before (no account needed).

Live dashboard for the 120-epoch run:
[wandb.ai/.../dit-fashionmnist](https://wandb.ai/gibson-yannick-czech-technical-university-in-prague/dit-fashionmnist/runs/3hj4jp48)

## Results

Trained 120 epochs on one A100. FID computed over 5,000 generated images vs.
the FashionMNIST test set (`uv run python ablation.py`).

**Classifier-free guidance scale** (50 sampling steps):

| cfg scale | FID |
|---|---|
| 1.0 | 92.34 |
| 2.0 | 74.48 |
| 4.0 | 70.93 |

**Sampling steps** (cfg = 2.0):

| Euler steps | FID |
|---|---|
| 10 | 75.53 |
| 50 | 74.07 |
| 250 | 80.44 |

**Takeaways**
- Classifier-free guidance helps monotonically - FID drops from 92.3
  (unguided) to 70.9 at scale 4.0.
- Sample quality is essentially flat from 10 to 50 steps and does *not* improve
  at 250. Flow matching's near-straight probability paths are well
  approximated by a coarse Euler integrator, so ~50 steps is the sweet spot -
  a concrete advantage over DDPM's hundreds of steps.
- Absolute FID is high because the Inception feature extractor expects RGB
  natural images, while these are grayscale clothing - a domain mismatch that
  inflates the score. The **relative trends** are the informative part.

## Scope notes

The original DiT runs in a VAE latent space on ImageNet; this is a smaller,
pixel-space, single-channel reproduction - FID is **not** directly comparable
to the paper. The backbone is intentionally small (depth 8, hidden 256) to fit
a one-day training run.

## License

[MIT](LICENSE) © 2026 Yannick Gibson
