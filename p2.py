import os
import random
import torch
import torchvision
import argparse
import numpy as np
import logging
from tqdm import tqdm
import argparse
from models import UNet_conditional


# setting seed
seed = 34820983
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample(self, model, n, labels, cfg_scale=3, denoise_time=1000):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, denoise_time)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--weight', type=str, default='diffusion_model.ckpt')
    # setting hyper parameters
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    cfg = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    model = UNet_conditional(num_classes=cfg.n_class).to(device)
    model.load_state_dict(torch.load(cfg.weight, map_location=device))
    model.eval()
    diffusion = Diffusion(img_size=cfg.img_size, device=device)

    sample = torch.ones(1000)
    for i in range(1000):
        sample[i] = i // 100
    sample = sample.long().to(device)
    for i in range(5):
        labels = sample[i*200: (i+1)*200]
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels, denoise_time=940)
    for j in range(100):
        idx = labels[0].type(torch.int).item()
        torchvision.utils.save_image(sampled_images[j], os.path.join(cfg.test_dir, f'{idx}_{(j+1):03d}.png'))
    for j in range(100):
        idx = labels[-1].type(torch.int).item()
        torchvision.utils.save_image(sampled_images[j+100], os.path.join(cfg.test_dir, f'{idx}_{(j+1):03d}.png'))
