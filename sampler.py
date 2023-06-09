import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised_first = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised_first) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised_second = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised_second) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def main():
    batch_size = 4
    network_pkl = 'alpha/6999.pkl'
    outdir='out'


    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)


    # Generate images.
    images = edm_sampler(net, latents)

    # Save images.
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    seeds = torch.arange(batch_size)
    sample_path = None
    for image_np, seed in tqdm.tqdm(zip(images_np, seeds)):
        os.makedirs(outdir, exist_ok=True)
        image_path = os.path.join(outdir, f'{seed:06d}.png')
        sample_path = image_path
        if image_np.shape[2] == 1:
            pic = image_np[:, :, 0]
            pic.dtype = np.uint8
            PIL.Image.fromarray(pic, 'L').save(image_path)
        else:
            PIL.Image.fromarray(image_np, 'RGB').save(image_path)
    print(f"Saved to {sample_path}")

if __name__ == "__main__":
    main()
