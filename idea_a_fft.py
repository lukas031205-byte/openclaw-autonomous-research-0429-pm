#!/usr/bin/env python3
"""Idea-A: FFT Low-Pass Filter Hypothesis
Mode-seeking decoder = low-pass filter → synthetic images lose MORE high-freq than natural
"""
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from diffusers.models import AutoencoderKL
import os, sys

# Setup
OUT_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-0429-pm"
os.makedirs(OUT_DIR, exist_ok=True)
device = "cpu"

# Load SD-VAE-ft-mse
print("Loading SD-VAE-ft-mse...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    cache_dir="/home/kas/.cache/huggingface/hub",
).to(device)
vae.eval()

# Load CIFAR-10 (100 images)
print("Loading CIFAR-10...")
cifar_transform = transforms.Compose([transforms.ToTensor()])
cifar_data = CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=cifar_transform)
cifar_batch = torch.stack([cifar_data[i][0] for i in range(100)]).to(device)  # [100, 3, 32, 32]
# Normalize to [-1,1]
cifar_batch = cifar_batch * 2 - 1

# Encode + decode CIFAR-10
print("VAE encode-decode CIFAR-10...")
with torch.no_grad():
    latents = vae.encode(cifar_batch).latent_dist.sample()  # [100, 4, 8, 8]
    recon_cifar = vae.decode(latents).sample  # [100, 3, 32, 32]
recon_cifar = torch.clamp(recon_cifar, -1, 1)

# FFT analysis on CIFAR-10 original vs recon
def high_freq_ratio(original, reconstructed, cutoff=0.7):
    """Fraction of high-freq power retained after recon. Higher = less filtering."""
    # 2D FFT on each image
    orig_fft = np.abs(np.fft.fft2(original.cpu().numpy()))
    recon_fft = np.abs(np.fft.fft2(reconstructed.cpu().numpy()))
    # Shift zero freq to center
    orig_fft = np.fft.fftshift(orig_fft)
    recon_fft = np.fft.fftshift(recon_fft)
    # Mask: low-freq center circle vs high-freq outer ring
    h, w = orig_fft.shape[-2:]
    y, x = np.ogrid[:h, :w]
    center_mask = ((x - w/2)**2 + (y - h/2)**2 <= (cutoff * min(h,w)/2)**2).astype(float)
    low_mask = center_mask
    high_mask = 1 - center_mask
    # HF power = sum of FFT magnitude in outer ring
    orig_hf = np.sum(orig_fft * high_mask, axis=(-1,-2))
    recon_hf = np.sum(recon_fft * high_mask, axis=(-1,-2))
    # Ratio
    ratio = recon_hf / (orig_hf + 1e-8)
    return ratio.mean(), ratio.std()

# Compute per-channel then average
# Use luminance: mean over RGB channels first
def luminance(x):
    return x.mean(dim=0, keepdim=True).mean(dim=0, keepdim=True)

cifar_orig_lum = cifar_batch.mean(dim=1, keepdim=True).squeeze(1)  # [100, 32, 32]
cifar_recon_lum = recon_cifar.mean(dim=1, keepdim=True).squeeze(1)
ratio_cifar, std_cifar = high_freq_ratio(cifar_orig_lum, cifar_recon_lum)

print(f"\n=== CIFAR-10 (SYNTHETIC) Results ===")
print(f"HF ratio: {ratio_cifar:.4f} ± {std_cifar:.4f}")
print(f"(Lower = more high-freq loss from decoder low-pass)")

# Save summary
with open(os.path.join(OUT_DIR, "idea_a_fft_results.txt"), "w") as f:
    f.write(f"Idea-A FFT Low-Pass Filter Hypothesis\n")
    f.write(f"CIFAR-10 synthetic: HF ratio = {ratio_cifar:.4f} ± {std_cifar:.4f}\n")
    f.write(f"(Lower ratio = decoder acts as stronger low-pass, more freq content destroyed)\n")
    f.write(f"\nFailure condition: if ratio ≈ 1.0, decoder doesn't selectively filter synthetic\n")

print(f"\nResults saved to {OUT_DIR}/idea_a_fft_results.txt")
print("Done.")