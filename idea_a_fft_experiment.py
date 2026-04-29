#!/usr/bin/env python3
"""
Idea-A FFT Low-Pass Hypothesis Experiment
Hypothesis: VAE encode-decode roundtrip = low-pass filter on image frequency content

Prediction:
- Synthetic images: high-frequency attenuation (ratio > 1)
- Natural images: minimal attenuation (ratio ≈ 1)
"""

import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from scipy import stats

# Paths
OUT_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-0429-pm"

# Config
N_IMAGES = 100
IMAGE_SIZE = 32
SEED = 42
DEVICE = "cpu"
DTYPE = torch.float32

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"=== Idea-A FFT Experiment ===")
print(f"N images per dataset: {N_IMAGES}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Device: {DEVICE}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
t0 = time.time()

# ─── Load SD-VAE ────────────────────────────────────────────────────────────────
print(f"\n[1] Loading SD-VAE (stabilityai/sd-vae-ft-mse)...")
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=DTYPE,
)
vae.eval()
vae.to(DEVICE)
print(f"VAE loaded. Latent scale factor: {vae.config.latent_channels}")
# For 32x32 RGB -> latent is (3, 4, 4) per channel or just flatten? 
# SD-VAE: 1 latent channel per RGB channel, factor=0.18215
# Latent spatial: 32/8 = 4, so (4,4) per channel

# ─── Load CIFAR-10 ─────────────────────────────────────────────────────────────
print(f"\n[2] Loading CIFAR-10 (train)...")
transform_cifar = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
cifar_data = datasets.CIFAR10(root=f"{OUT_DIR}/data_cifar10", train=True, download=True, transform=transform_cifar)
cifar_subset = torch.utils.data.Subset(cifar_data, random.sample(range(len(cifar_data)), N_IMAGES))
cifar_loader = DataLoader(cifar_subset, batch_size=N_IMAGES, shuffle=False)

for imgs, _ in cifar_loader:
    cifar_images = imgs  # (N, 3, 32, 32)
print(f"CIFAR-10 loaded: {cifar_images.shape}")

# ─── Load COCO Val ─────────────────────────────────────────────────────────────
print(f"\n[3] Loading COCO Val2017 (100 images)...")
coco_root = f"{OUT_DIR}/data_coco"
transform_coco = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Try to load from cache or download
try:
    # Check if we have a cached COCO dataset
    from torchvision.datasets import CocoDetection
    import os
    val_ann = f"{coco_root}/annotations/instances_val2017.json"
    val_img = f"{coco_root}/val2017"
    if os.path.exists(val_ann):
        coco_ds = CocoDetection(root=val_img, annFile=val_ann, transform=transform_coco)
        # Sample 100
        indices = random.sample(range(len(coco_ds)), min(N_IMAGES, len(coco_ds)))
        coco_images = torch.stack([coco_ds[i][0] for i in indices])
    else:
        raise FileNotFoundError("COCO annotations not found")
except Exception as e:
    print(f"COCO detection failed ({e}), trying alternative download...")
    # Fallback: use a simpler approach
    try:
        import subprocess
        subprocess.run([
            "python3", "-c",
            f"from torchvision.datasets import CocoDetection; d=CocoDetection('{coco_root}/val2017', '{coco_root}/annotations/instances_val2017.json'); print(len(d))"
        ], capture_output=True, timeout=30)
    except:
        pass
    
    # Try to just use CIFAR-100 or STL-10 as natural image proxy
    # Actually let's try to download COCO properly
    try:
        from pycocotools.coco import COCO
        import urllib.request
        import zipfile
        
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        img_url = "http://images.cocodataset.org/zips/val2017.zip"
        
        os.makedirs(f"{coco_root}/annotations", exist_ok=True)
        os.makedirs(f"{coco_root}/val2017", exist_ok=True)
        
        print("Attempting COCO Val2017 download...")
        # Download annotations
        if not os.path.exists(val_ann):
            print("Downloading COCO annotations...")
            urllib.request.urlretrieve(ann_url, f"{coco_root}/annotations/ann.zip")
            with zipfile.ZipFile(f"{coco_root}/annotations/ann.zip", 'r') as z:
                z.extractall(f"{coco_root}/annotations/")
        
        if not os.path.exists(val_img + "/000000000000.jpg"):
            print("Downloading COCO val images (~700MB)...")
            urllib.request.urlretrieve(img_url, f"{coco_root}/val2017.zip")
            with zipfile.ZipFile(f"{coco_root}/val2017.zip", 'r') as z:
                z.extractall(coco_root)
        
        coco_ds = CocoDetection(root=val_img, annFile=val_ann, transform=transform_coco)
        indices = random.sample(range(len(coco_ds)), min(N_IMAGES, len(coco_ds)))
        coco_images = torch.stack([coco_ds[i][0] for i in indices])
    except Exception as e2:
        print(f"COCO download failed: {e2}")
        print("Falling back to STL-10 as natural image proxy...")
        stl_data = datasets.STL10(root=f"{OUT_DIR}/data_stl10", split='train', download=True, transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]))
        stl_subset = torch.utils.data.Subset(stl_data, random.sample(range(len(stl_data)), N_IMAGES))
        stl_loader = DataLoader(stl_subset, batch_size=N_IMAGES, shuffle=False)
        for imgs, _ in stl_loader:
            coco_images = imgs  # (N, 3, 32, 32)
        print(f"STL-10 loaded as natural proxy: {coco_images.shape}")

print(f"Natural images loaded: {coco_images.shape}")

# ─── VAE Encode-Decode ──────────────────────────────────────────────────────────
print(f"\n[4] Running VAE encode-decode roundtrip...")

def vae_roundtrip(images):
    """Encode images through VAE and decode back."""
    with torch.no_grad():
        # Scale from [0,1] to VAE expected range
        x = images.to(DTYPE).to(DEVICE)
        # VAE encode
        latent = vae.encode(x).latent_dist.sample()
        # Decode
        recon = vae.decode(latent).sample
    return recon.cpu()

cifar_recon = vae_roundtrip(cifar_images)
print(f"CIFAR reconstructions: {cifar_recon.shape}")

coco_recon = vae_roundtrip(coco_images)
print(f"COCO reconstructions: {coco_recon.shape}")

# ─── FFT Analysis ──────────────────────────────────────────────────────────────
print(f"\n[5] Computing 2D FFT and high-frequency energy...")

def compute_hf_energy(img_tensor):
    """
    Compute high-frequency energy of a 2D image.
    Use 2D FFT, compute radial frequency mask.
    High freq = outer 50% of frequency plane (wavenumber > median).
    Returns total energy in high-freq region.
    """
    # img_tensor: (3, H, W) or (H, W)
    if img_tensor.dim() == 3:
        # Average across channels for gray equivalent
        img = img_tensor.mean(dim=0).numpy()
    else:
        img = img_tensor.numpy()
    
    H, W = img.shape
    
    # 2D FFT
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) ** 2
    
    # Radial frequency coordinates (centered)
    u = np.arange(H) - H // 2
    v = np.arange(W) - W // 2
    U, V = np.meshgrid(v, u)
    R = np.sqrt(U**2 + V**2)
    
    # Median frequency as cutoff
    R_flat = R.ravel()
    mag_flat = magnitude.ravel()
    cutoff = np.percentile(R_flat, 50)  # outer 50% = high freq
    
    # High frequency mask
    hf_mask = R > cutoff
    
    E_total = mag_flat.sum()
    E_hf = (mag_flat * hf_mask).sum()
    
    return float(E_hf), float(E_total)

def compute_kurtosis(img_tensor):
    """Compute kurtosis of image pixel values as sanity check."""
    if img_tensor.dim() == 3:
        img = img_tensor.mean(dim=0).numpy()
    else:
        img = img_tensor.numpy()
    img_flat = img.ravel()
    # Excess kurtosis: Kurtosis - 3 (0 for Gaussian)
    k = stats.kurtosis(img_flat, fisher=True)
    return float(k)

# Compute per-image metrics
print("Processing CIFAR-10...")
cifar_results = []
for i in range(N_IMAGES):
    orig = cifar_images[i]
    recon = cifar_recon[i]
    
    E_hf_orig, _ = compute_hf_energy(orig)
    E_hf_recon, _ = compute_hf_energy(recon)
    
    hf_ratio = E_hf_orig / E_hf_recon if E_hf_recon > 0 else np.nan
    
    kurt_orig = compute_kurtosis(orig)
    kurt_recon = compute_kurtosis(recon)
    
    cifar_results.append({
        "image_idx": i,
        "hf_ratio": hf_ratio,
        "kurt_orig": kurt_orig,
        "kurt_recon": kurt_recon,
    })

print("Processing COCO (natural)...")
coco_results = []
for i in range(N_IMAGES):
    orig = coco_images[i]
    recon = coco_recon[i]
    
    E_hf_orig, _ = compute_hf_energy(orig)
    E_hf_recon, _ = compute_hf_energy(recon)
    
    hf_ratio = E_hf_orig / E_hf_recon if E_hf_recon > 0 else np.nan
    
    kurt_orig = compute_kurtosis(orig)
    kurt_recon = compute_kurtosis(recon)
    
    coco_results.append({
        "image_idx": i,
        "hf_ratio": hf_ratio,
        "kurt_orig": kurt_orig,
        "kurt_recon": kurt_recon,
    })

# ─── Summary Statistics ────────────────────────────────────────────────────────
print(f"\n[6] Computing summary statistics...")

cifar_hf_ratios = [r["hf_ratio"] for r in cifar_results if not np.isnan(r["hf_ratio"])]
coco_hf_ratios = [r["hf_ratio"] for r in coco_results if not np.isnan(r["hf_ratio"])]

cifar_kurt_ratios = [r["kurt_recon"]/r["kurt_orig"] if r["kurt_orig"] != 0 else np.nan for r in cifar_results]
coco_kurt_ratios = [r["kurt_recon"]/r["kurt_orig"] if r["kurt_orig"] != 0 else np.nan for r in coco_results]

cifar_kurt_ratios = [x for x in cifar_kurt_ratios if not np.isnan(x)]
coco_kurt_ratios = [x for x in coco_kurt_ratios if not np.isnan(x)]

# T-test: is CIFAR ratio > COCO ratio?
cifar_arr = np.array(cifar_hf_ratios)
coco_arr = np.array(coco_hf_ratios)
t_stat, p_val = stats.ttest_ind(cifar_arr, coco_arr)

# One-sided test: is CIFAR ratio > 1? (decoder destroys high-freq)
cifar_t_stat_one, cifar_p_one = stats.ttest_1samp(cifar_arr, 1.0)

# Kurtosis t-test
cifar_kurt_arr = np.array(cifar_kurt_ratios)
coco_kurt_arr = np.array(coco_kurt_ratios)
kurt_t_stat, kurt_p_val = stats.ttest_ind(cifar_kurt_arr, coco_kurt_arr)

results = {
    "hypothesis": "VAE encode-decode roundtrip = low-pass filter on image frequency content",
    "config": {
        "n_images_per_dataset": N_IMAGES,
        "image_size": IMAGE_SIZE,
        "vae_model": "stabilityai/sd-vae-ft-mse",
        "dtype": str(DTYPE),
        "device": DEVICE,
        "seed": SEED,
    },
    "cifar10_synthetic": {
        "mean_hf_ratio": float(np.mean(cifar_arr)),
        "std_hf_ratio": float(np.std(cifar_arr)),
        "median_hf_ratio": float(np.median(cifar_arr)),
        "min_hf_ratio": float(np.min(cifar_arr)),
        "max_hf_ratio": float(np.max(cifar_arr)),
        "n_valid": len(cifar_arr),
        "hf_ratio_t_one_sided_p": float(cifar_p_one / 2) if cifar_t_stat_one > 0 else float(1 - cifar_p_one / 2),
        "per_image": cifar_results,
    },
    "coco_natural": {
        "mean_hf_ratio": float(np.mean(coco_arr)),
        "std_hf_ratio": float(np.std(coco_arr)),
        "median_hf_ratio": float(np.median(coco_arr)),
        "min_hf_ratio": float(np.min(coco_arr)),
        "max_hf_ratio": float(np.max(coco_arr)),
        "n_valid": len(coco_arr),
        "per_image": coco_results,
    },
    "statistical_tests": {
        "cifar_vs_coco_hf_ttest_t": float(t_stat),
        "cifar_vs_coco_hf_ttest_p": float(p_val),
        "cifar_hf_ratio_one_sided_p": float(cifar_p_one / 2) if cifar_t_stat_one > 0 else float(1 - cifar_p_one / 2),
        "kurtosis_ratio_cifar_vs_coco_t": float(kurt_t_stat),
        "kurtosis_ratio_cifar_vs_coco_p": float(kurt_p_val),
    },
    "interpretation": {
        "if_mean_hf_ratio_cifar_gt_1_and_coco_approx_1": "SUPPORTS hypothesis: decoder attenuates high-freq more on synthetic",
        "if_mean_hf_ratio_both_approx_1": "FAILS hypothesis: decoder does not act as low-pass",
        "if_mean_hf_ratio_cifar_lt_1": "NEGATIVE RESULT: decoder ENHANCES high-freq on synthetic",
        "actual_cifar_mean_hf_ratio": float(np.mean(cifar_arr)),
        "actual_coco_mean_hf_ratio": float(np.mean(coco_arr)),
        "actual_cifar_kurtosis_ratio_mean": float(np.mean(cifar_kurt_arr)),
        "actual_coco_kurtosis_ratio_mean": float(np.mean(coco_kurt_arr)),
    },
    "runtime_seconds": time.time() - t0,
    "completion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# ─── Save Results ──────────────────────────────────────────────────────────────
out_path = f"{OUT_DIR}/idea_a_fft_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

elapsed = time.time() - t0
print(f"\n=== RESULTS ===")
print(f"CIFAR-10 (synthetic) mean HF ratio: {np.mean(cifar_arr):.4f} ± {np.std(cifar_arr):.4f}")
print(f"COCO (natural) mean HF ratio: {np.mean(coco_arr):.4f} ± {np.std(coco_arr):.4f}")
print(f"CIFAR vs COCO t-test: t={t_stat:.4f}, p={p_val:.6f}")
print(f"CIFAR HF ratio one-sided p (vs 1.0): {cifar_p_one/2 if cifar_t_stat_one > 0 else 1 - cifar_p_one/2:.6f}")
print(f"CIFAR-10 mean kurtosis ratio: {np.mean(cifar_kurt_arr):.4f}")
print(f"COCO mean kurtosis ratio: {np.mean(coco_kurt_arr):.4f}")
print(f"Total runtime: {elapsed:.1f}s")
print(f"Results saved to: {out_path}")

# Interpretation
mean_cifar = np.mean(cifar_arr)
mean_coco = np.mean(coco_arr)
print(f"\n=== INTERPRETATION ===")
if mean_cifar > 1.1 and mean_coco < 1.05:
    print("SUPPORTS hypothesis: VAE decoder acts as low-pass filter on synthetic > natural")
elif mean_cifar < 0.95:
    print("NEGATIVE RESULT: Decoder ENHANCES high-frequency on synthetic (ratio < 1)")
else:
    print(f"INCONCLUSIVE: CIFAR={mean_cifar:.4f}, COCO={mean_coco:.4f} (neither strongly >1 nor ≈1)")