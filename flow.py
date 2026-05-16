"""Conditional flow matching: training loss + ODE sampler.

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023).

Path:    x_t = (1 - t) * noise + t * data,   t in [0, 1]
Target:  velocity v* = data - noise   (constant along the linear path)
Loss:    MSE between the network's predicted velocity and v*.

Sampling integrates dx/dt = v(x, t) from t=0 (noise) to t=1 (data).
"""

import torch


def flow_matching_loss(model, x_data, y):
    """One training step's loss for a batch of real images x_data."""
    b = x_data.shape[0]
    t = torch.rand(b, device=x_data.device)
    noise = torch.randn_like(x_data)
    t_b = t.view(b, 1, 1, 1)
    x_t = (1 - t_b) * noise + t_b * x_data
    target = x_data - noise
    pred = model(x_t, t, y, train=True)
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def sample(model, n, class_labels, num_steps=50, cfg_scale=2.0, device="cuda"):
    """Generate `n` images via Euler integration of the velocity field.

    Channel count and resolution are taken from the model, so this works for
    both grayscale FashionMNIST and RGB CIFAR-10. cfg_scale = 1.0 disables
    classifier-free guidance.
    """
    model.eval()
    x = torch.randn(n, model.in_ch, model.img_size, model.img_size, device=device)
    null = torch.full_like(class_labels, model.y_embed.num_classes)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((n,), i * dt, device=device)
        if cfg_scale != 1.0:
            v = model(torch.cat([x, x]), torch.cat([t, t]),
                      torch.cat([class_labels, null]))
            v_cond, v_uncond = v.chunk(2)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = model(x, t, class_labels)
        x = x + v * dt

    return x.clamp(-1, 1)
