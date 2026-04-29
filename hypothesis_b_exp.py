#!/usr/bin/env python3
"""
Hypothesis B: Decoder as Confidence-Dependent Structure Reshaper
Latent Neighborhood Coherence vs HF Ratio on CIFAR-10 (memory-safe version)
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from diffusers import AutoencoderKL
import numpy as np
from tqdm import tqdm
import os, json, gc

DEVICE = "cpu"
N_IMAGES = 1000
K_NEIGHBORS = 20
BATCH_SIZE = 50
LATENT_SIZE = 16  # image resized to 128x128 -> latent 2x16x16
EPS = 1e-6

OUT_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-0429-pm"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# --- 1. Load CIFAR-10 (1000 images, resized to 128x128) ---
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Collect images
all_images = []
for imgs, _ in loader:
    all_images.append(imgs)
    if len(all_images) * BATCH_SIZE >= N_IMAGES:
        break
images = torch.cat(all_images)[:N_IMAGES].to(DEVICE)
print(f"Loaded {len(images)} CIFAR-10 images, shape {images.shape}")

# --- 2. Load VAE ---
print("Loading SD-VAE-ft-mse...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_device=DEVICE,
)
vae.eval()
print("VAE loaded.")

# --- 3. Encode in batches ---
@torch.no_grad()
def encode_batch(imgs):
    latent = vae.encode(imgs).latent_dist.sample()
    # latent: [B, 4, H, W] where H=16, W=16 for 128x128 input
    return latent

print("Encoding images to latent space in batches...")
all_latents = []
for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Encoding"):
    batch = images[i:i+BATCH_SIZE]
    lat = encode_batch(batch)
    all_latents.append(lat.cpu())
    del lat
    gc.collect()

latents = torch.cat(all_latents, dim=0)  # [N, 4, 16, 16]
print(f"Latent shape: {latents.shape}")

B, C, H, W = latents.shape
latents_flat = latents.reshape(B, -1).float()  # [B, 4*16*16]
print(f"Flat latent dim: {latents_flat.shape[1]}")

# --- 4. Compute k-NN distances & coherence ---
print(f"\nComputing {K_NEIGHBORS}-NN distances for {B} images...")

dists = torch.cdist(latents_flat, latents_flat, p=2)
dists[:, torch.arange(B)] = float('inf')

knn_values, _ = torch.topk(dists, k=K_NEIGHBORS, largest=False, dim=1)
mean_knn_dist = knn_values.mean(dim=1)  # [B]
coherence = 1.0 / (mean_knn_dist + EPS)  # [B]

print(f"Mean knn distance: {mean_knn_dist.mean().item():.4f}")
print(f"Coherence: min={coherence.min().item():.4f}, max={coherence.max().item():.4f}, mean={coherence.mean().item():.4f}")

# --- 5. Compute HF ratio per image ---
@torch.no_grad()
def compute_hf_ratio_batch(imgs_batch):
    """HF ratio per image in batch."""
    import torch.fft as fft
    B, C, H, W = imgs_batch.shape
    
    fft_orig = fft.fft2(imgs_batch)
    fft_orig_mag = torch.abs(fft_orig)
    cx, cy = W // 2, H // 2
    r = min(W, H) // 8
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = ((x - cx)**2 + (y - cy)**2).float()
    mask = (dist > r**2).float()
    mask = mask.unsqueeze(0).unsqueeze(0).expand(C, H, W).clone()
    
    hf_orig = (fft_orig_mag * mask.unsqueeze(1)).sum(dim=(1,2,3)) / (C * H * W)
    
    # Decode batch
    lat_batch = vae.encode(imgs_batch).latent_dist.sample()
    rec_batch = vae.decode(lat_batch).sample
    rec_batch = torch.nn.functional.interpolate(rec_batch, size=(H, W), mode='bilinear', align_corners=False)
    
    fft_rec = fft.fft2(rec_batch)
    fft_rec_mag = torch.abs(fft_rec)
    hf_rec = (fft_rec_mag * mask.unsqueeze(1)).sum(dim=(1,2,3)) / (C * H * W)
    
    ratios = hf_rec / (hf_orig + EPS)
    return ratios.cpu()

print("\nComputing HF ratios per image...")
hf_ratios = []
for i in tqdm(range(0, B, BATCH_SIZE), desc="HF ratio"):
    batch = images[i:i+BATCH_SIZE].to(DEVICE)
    ratios = compute_hf_ratio_batch(batch)
    hf_ratios.append(ratios)
    del batch
    gc.collect()

hf_ratios = torch.cat(hf_ratios, dim=0)
print(f"HF ratio stats: min={hf_ratios.min().item():.4f}, mean={hf_ratios.mean().item():.4f}, max={hf_ratios.max().item():.4f}")

# --- 6. Split into high/low coherence ---
coherence_np = coherence.cpu().numpy()
hf_np = hf_ratios.cpu().numpy()

threshold = np.median(coherence_np)
high_coh_mask = coherence_np >= threshold
low_coh_mask = coherence_np < threshold

high_coh_hf = hf_np[high_coh_mask]
low_coh_hf = hf_np[low_coh_mask]

print(f"\n=== Results ===")
print(f"Threshold (median coherence): {threshold:.6f}")
print(f"High-coherence subset: n={high_coh_mask.sum()}, HF ratio mean={high_coh_hf.mean():.4f}, std={high_coh_hf.std():.4f}")
print(f"Low-coherence subset:  n={low_coh_mask.sum()}, HF ratio mean={low_coh_hf.mean():.4f}, std={low_coh_hf.std():.4f}")

from scipy.stats import spearmanr
corr, pval = spearmanr(coherence_np, hf_np)
print(f"Spearman r(coherence, HF ratio): {corr:.4f}, p={pval:.4f}")

# --- 7. Report ---
results = {
    "n_images": N_IMAGES,
    "k_neighbors": K_NEIGHBORS,
    "vae_model": "stabilityai/sd-vae-ft-mse",
    "image_size": 128,
    "latent_shape": [C, H, W],
    "latent_dim": int(C * H * W),
    "mean_knn_distance": float(mean_knn_dist.mean().item()),
    "coherence_min": float(coherence.min().item()),
    "coherence_max": float(coherence.max().item()),
    "coherence_mean": float(coherence.mean().item()),
    "coherence_median": float(np.median(coherence_np)),
    "hf_ratio_mean": float(hf_ratios.mean().item()),
    "hf_ratio_std": float(hf_ratios.std().item()),
    "hf_ratio_min": float(hf_ratios.min().item()),
    "hf_ratio_max": float(hf_ratios.max().item()),
    "high_coherence_n": int(high_coh_mask.sum()),
    "high_coherence_hf_mean": float(high_coh_hf.mean()),
    "high_coherence_hf_std": float(high_coh_hf.std()),
    "low_coherence_n": int(low_coh_mask.sum()),
    "low_coherence_hf_mean": float(low_coh_hf.mean()),
    "low_coherence_hf_std": float(low_coh_hf.std()),
    "spearman_r": float(corr),
    "spearman_p": float(pval),
    "hypothesis_b_prediction": "high coherence → HF ratio > 1 (decoder enhances HF); low coherence → HF ratio < 1 (decoder blurs)",
    "hypothesis_b_match": bool(high_coh_hf.mean() > low_coh_hf.mean()),
    "interpretation": "If high-coherence images have HF ratio > 1 and low-coherence < 1, hypothesis B is supported",
}

out_path = os.path.join(OUT_DIR, "hypothesis_b_latent_coherence_results.txt")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_path}")
print(json.dumps(results, indent=2))