from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from unet import ContextUnet
from edm import EDMPrecond as EDM_Unet
from losses import *
import os
from math import log

# empty cache, just in case
torch.cuda.empty_cache()

class DDPM2(nn.Module):
    def __init__(self, nn_model, n_T, device, data_sigma=0.5, drop_prob=0.1, rho=7, **kwargs):
        super(DDPM2, self).__init__()

        self.nn_model = nn_model
        self.n_T = n_T
        self.device = device
        self.data_sigma = data_sigma

        self.rho = rho  # Weighting of sigma; rho = 7 in the paper, range 5-10 considered valid; 
                        # more -> more effort during low noise; 3 - equalizes truncation error
        self.drop_prob = drop_prob

        self.sigma_max = 80
        self.sigma_min = 0.002

        self.P_mean = -1.2
        self.P_std = 1.2

        self.loss = EDMLoss()

        self.precision = torch.float16

    def denoise(self, x, sigma, *args, **kwargs):
#        c_skip = self.data_sigma ** 2 / (sigma ** 2 + self.data_sigma ** 2)
#        c_out = sigma * self.data_sigma / (sigma ** 2 + self.data_sigma ** 2).sqrt()
#        c_in = 1 / (self.data_sigma ** 2 + sigma ** 2).sqrt()
#        c_noise = sigma.log() / 4

        zeros = torch.zeros((x.shape[0],)).to(torch.int64).to(self.device)
#        F_x = self.nn_model(c_in * x, zeros, c_noise.flatten(), zeros)
#        D_x = c_skip * x + c_out * F_x
#        return D_x
        return self.nn_model(x, zeros, sigma, zeros) * sigma + x
#        return self.nn_model(x, sigma)

    def forward(self, x, c):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=self.device)
#        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = torch.rand(x.shape[0], 1, 1, 1) * (log(self.sigma_max) - log(self.sigma_min)) + self.sigma_min
        sigma = sigma.exp().to(self.device)
        weight = 1 / sigma ** 2
        n = torch.randn_like(x) * sigma
        D_theta = self.denoise(x + n, sigma)
        loss = weight * ((D_theta - x) ** 2)
        return loss.mean()
        #return self.loss(self.denoise, x, None, None).mean()

    def sample(self, n_sample, size, *args, **kwargs):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = self.sigma_min
        sigma_max = self.sigma_max
        latents = torch.stack([torch.randn(size, device=self.device) for _ in range(n_sample)])
        rho = self.rho
        net = self.nn_model
        num_steps = self.n_T

        S_churn = 20
        S_min = 0
        S_max = 50
        S_noise = 1.003

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=self.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        x_i_store = []

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.denoise(x_hat, t_hat)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x_next, t_next)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            x_i_store.append(denoised.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)

        return x_next, x_i_store

def train_mnist():
    # hardcoding these here
    n_epoch = 5
    batch_size = 64 # 128 was default
    n_T = 100 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './output-3/'
    ws_test = [1.0]#[0.0, 0.5, 2.0] # strength of generative guidance

    # create the directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #model = EDM_Unet(28, 1, label_dim=0, use_fp16=False, sigma_min=0.002, sigma_max=80, sigma_data=0.3)
    #ddpm = DDPM2(model, data_sigma=0.3, rho=7, device=device, n_T=n_T)
    model = ContextUnet(in_channels=1, n_feat=n_feat)
    ddpm = DDPM2(nn_model=model, data_sigma=0.3, rho=7, n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.AdamW(ddpm.parameters(), lr=lrate, betas=[0.9,0.999], eps=1e-8)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%10==9 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()
