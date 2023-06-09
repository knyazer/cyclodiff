import os
import tqdm
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import wandb

wandb.init()

from torch_utils.misc import InfiniteSampler
from networks import EDMPrecond
from dataset import ImageFolderDataset
from losses import EDMLoss
from sampler import edm_sampler

def main():

    batch_size = 12
    train_for_kimg = 1000
    lr = 5e-4
    img_dim = 64
    img_chn = 3


    seed = 42
    ema_halflife_kimg = 200
    path_to_images = 'summer2winter/trainA'
    version = '/ml/cyclodiff/alpha'

    ema_rampup_ratio = 0.05

    if not os.path.exists(f'{version}'):
        os.makedirs(f'{version}')

    device = torch.device('cuda:0')
    torch.manual_seed(seed)

    model = EDMPrecond( img_resolution=img_dim, 
                        img_channels=img_chn,
                        use_fp16=True,
                        sigma_data=0.5,
                        model_type='SongUNet',
                        embedding_type='positional',
                        encoder_type='standard',
                        decoder_type='standard').to(device)

    clip_value = 1e2
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = EDMLoss()

    dataset_obj = ImageFolderDataset(path_to_images, force_size=img_dim)
    dataset_sampler = InfiniteSampler(dataset=dataset_obj, rank=0, num_replicas=1, seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_size))

    ema_loss = None
    cur_nimg = 0
    st = time.time()

    for it in tqdm.tqdm(range(train_for_kimg * 1000 // batch_size)):
        images, _ = next(dataset_iterator)
        images = images.to(device).to(torch.float32) / 127.5 - 1

        optimizer.zero_grad(set_to_none=True)

        loss = loss_fn(net=model, images=images)
        loss = loss.sum().mul(1 / batch_size)

        if it % 10 == 9:
            wandb.log({'loss': loss.item() / (img_dim * img_dim)}, step=int((time.time() - st) * 10))

        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=clip_value, neginf=-clip_value, out=param.grad)

        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema_model.parameters(), model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        cur_nimg += batch_size

        if it % 1000 == 999:
            data = dict(ema=ema_model, loss_fn=loss_fn, augment_pipe=None, dataset_kwargs=None)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    data[key] = value.cpu()
                del value # conserve memory

            with open(f'{version}/{it}.pkl', 'wb') as f:
                pickle.dump(data, f)

            del data # conserve memory

            # sample a bunch of images just for funzies
            with torch.no_grad():
                pics = edm_sampler(net=ema_model, latents=torch.randn([2, img_chn, img_dim, img_dim]).to(device))
                pics = (pics * 127.5 + 127.5).permute(0,2,3,1).clip(0, 255).cpu().numpy().astype(np.uint8)
                wandb.log({'images': [wandb.Image(pic) for pic in pics]})


    torch.save(model, f'{version}/model.pkl')
    print('Done')

if __name__ == '__main__':
    main()
