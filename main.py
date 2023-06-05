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
import os

# empty cache, just in case
torch.cuda.empty_cache()

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


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

    def c_in(self, sigma):
        return 1.0 / torch.sqrt(sigma ** 2 + self.data_sigma ** 2)

    def c_out(self, sigma):
        return sigma * self.data_sigma / torch.sqrt(sigma ** 2 + self.data_sigma ** 2)

    def c_skip(self, sigma):
        return (self.data_sigma ** 2) / (sigma ** 2 + self.data_sigma ** 2)

    def c_noise(self, sigma):
        return torch.log(sigma) / 4.0 # mmmm, smart formula (its empirical, probs should find something better)

    def loss_weighting(self, sigma):
        return (self.data_sigma ** 2 + sigma ** 2) / ((self.data_sigma * sigma) ** 2)

    # Sigma scheduler for sampling
    def compute_sigma(self, t):
        res = (
            (self.sigma_max ** (1 / self.rho)) +
            (t / (self.n_T - 1)) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        )
        res[t == self.n_T] = 0.0

        return res ** self.rho

    def forward(self, x, c):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=self.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        pred = self.nn_model(x + torch.rand_like(x) * sigma, sigma)
        loss = self.loss_weighting(sigma) * ((pred - x) ** 2)
        return loss.mean()

    def sample(self, n_sample, size, *args, **kwargs):
        step_indices = torch.arange(self.n_T, dtype=torch.float32, device=self.device)
        sigma = self.compute_sigma(step_indices)
        latents = torch.randn(n_sample, *size, device=self.device)

        S_churn = 0.0
        S_max = float('inf')
        S_noise = 1.0
        S_min = 0.0

        x_i_store = []

        x_next = latents * sigma[0]
        print("Sampling...")
        for i, (t_cur, t_next) in tqdm(enumerate(zip(sigma[:-1], sigma[1:]))): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / self.n_T, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.nn_model(x_hat, t_hat)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.n_T - 1:
                denoised = self.nn_model(x_next, t_next)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            if i%10==0 or i==self.n_T or i < 8:
                x_i_store.append(x_next.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)

        return x_next, x_i_store


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)

        """
        We are trying to train kind of diffusion gan
        Lets denote x_t as the diffusion process on domain A, and y_t as the diffusion process on domain B
        
        Firstly, we make a prediction of x_t noise, conditioned on the y_t=y0*sqrt(alpha_t) + sqrt(1-alpha_t)*eps
        Then, we make a prediction of y_{t-1} noise, conditioned on the x_t with removed step noise, so x_{t-1}_est = x_t - sqrt(alpha_t)*eps, kind of like single sampling step
        Hence we get noise for y_{t-1}, which we can evaluate explicitly

        """
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

def train_mnist():
    # hardcoding these here
    n_epoch = 20
    batch_size = 32 # 128 was default
    n_T = 200 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 64 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './output-2/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    # create the directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ddpm = DDPM2(nn_model=EDM_Unet(img_channels=1, img_resolution=28, sigma_data=0.5), rho=7, n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

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

                if ep%5==0 or ep == int(n_epoch-1):
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
