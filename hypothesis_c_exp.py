"""
Hypothesis C: Semantic Attribute Erasure via Latent Space Interpolation Uncertainty
Kernel experiment for Domain/Scalpel

Prediction: Moving latent toward training prototype (α → 0) should REDUCE CLIP semantic drift
  → CLIP similarity should INCREASE as α decreases toward 0.

Setup:
- Training set (prototypes): 5000 random CIFAR-10 train images
- Test set (natural): 500 CIFAR-10 test images
- Use SD-VAE-ft-mse to encode all images (resize 32x32 → 512x512)
- Nearest neighbor in latent space (cosine distance)
- Interpolate: z_α = α·z + (1-α)·z*
- Decode and measure CLIP similarity with original
"""

import torch
import clip
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from diffusers.models import AutoencoderKL
import time
import os
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_N = 5000       # size of "training set" (prototype pool)
TEST_N  = 500        # number of test images to evaluate
ALPHAS  = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
BATCH_SIZE = 8       # decode batch size
SEED = 42
OUT_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-0429-pm"

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"DEVICE={DEVICE}")
print(f"TRAIN_N={TRAIN_N}, TEST_N={TEST_N}")
print(f"ALPHAS={ALPHAS}")
print(f"sd-vae-ft-mse + openai/clip-vit-b/32")

# ── 1. Load CLIP ──────────────────────────────────────────────────────────────
print("\n[1] Loading CLIP ViT-B/32...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
# image encode helper
def clip_encode(img_tensor):
    """img_tensor: normalized [0,1], shape [B,3,H,W]"""
    with torch.no_grad():
        emb = clip_model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

# ── 2. Load SD-VAE ────────────────────────────────────────────────────────────
print("\n[2] Loading SD-VAE-ft-mse...")
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    local_files_only=False,
)
vae = vae.to(DEVICE).eval()

# ── 3. Load CIFAR-10 ──────────────────────────────────────────────────────────
print("\n[3] Loading CIFAR-10...")
transform_to_pil = transforms.ToPIL()
resize_to_512 = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
])

# Normalize like SD: [0,1] → encode expects mean=0, std=0.5? Actually SD-VAE was trained on latent space
# We'll encode with the VAE encoder which handles the scaling internally
# The diffusers AutoencoderKL expects pixel values in [-1, 1] when using decode
# But encode accepts [0,1]?

# Actually: AutoencoderKL encode -> expects pixel values normalized. Let's just use [0,1] which is what our tensors are
# The VAE was trained with latent diffusion, it handles the scaling internally

cifar_train = datasets.CIFAR10(root='~/data/cifar10', train=True, download=True)
cifar_test  = datasets.CIFAR10(root='~/data/cifar10', train=False, download=True)

# Sample train indices (prototypes)
train_idx = np.random.permutation(len(cifar_train))[:TRAIN_N].tolist()
train_images = [cifar_train[i][0] for i in train_idx]
# Sample test indices (natural images = test set)
test_idx = np.random.permutation(len(cifar_test))[:TEST_N].tolist()
test_images = [cifar_test[i][0] for i in test_idx]

print(f"  Train (prototypes): {len(train_images)} images")
print(f"  Test  (natural):   {len(test_images)} images")

# ── 4. Encode all images to latents ──────────────────────────────────────────
def encode_images(images, desc="encoding"):
    """images: list of PIL images. Returns list of latent tensors on CPU."""
    latents = []
    for i in tqdm(range(0, len(images), BATCH_SIZE), desc=desc):
        batch_pil = images[i:i+BATCH_SIZE]
        batch_tensor = torch.stack([resize_to_512(img) for img in batch_pil]).to(DEVICE)
        with torch.no_grad():
            # AutoencoderKL encode returns latent distribution (mean + logvar), we use .latent_dist.mean
            dist = vae.encode(batch_tensor).latent_dist
            z = dist.mean.float()
        # Move to CPU
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)

print("\n[4a] Encoding training set (prototype latents)...")
t0 = time.time()
z_train = encode_images(train_images, "train encode")
print(f"  done in {time.time()-t0:.1f}s, shape={z_train.shape}")  # [N, 4, 64, 64]

print("\n[4b] Encoding test set (natural image latents)...")
t0 = time.time()
z_test = encode_images(test_images, "test encode")
print(f"  done in {time.time()-t0:.1f}s, shape={z_test.shape}")

# Flatten latents for distance computation
def flatten_latents(z):
    """z: [N, 4, 64, 64] -> [N, D]"""
    return z.flatten(start_dim=1)

z_train_flat = flatten_latents(z_train)
z_test_flat  = flatten_latents(z_test)

# Normalize for cosine similarity
z_train_flat = z_train_flat / z_train_flat.norm(dim=-1, keepdim=True)
z_test_flat  = z_test_flat  / z_test_flat.norm(dim=-1, keepdim=True)

# ── 5. Find nearest prototype for each test image ───────────────────────────
print("\n[5] Finding nearest prototype for each test image (cosine)...")
# cosine sim = dot product since normalized
sim_matrix = torch.mm(z_test_flat, z_train_flat.T)  # [test_n, train_n]
nearest_idx = sim_matrix.argmax(dim=1).numpy()        # [test_n]
z_prototypes = z_train[nearest_idx]                   # [test_n, 4, 64, 64]
print(f"  done. nearest prototype indices shape: {nearest_idx.shape}")

# ── 6. CLIP prep: prepare precomputed clip embeddings for original images ────
print("\n[6] Computing CLIP embeddings for original test images...")
# Get CLIP inputs for original images
clip_inputs = []
for img in tqdm(test_images, desc="CLIP preprocess"):
    t = clip_preprocess(img).to(DEVICE)
    clip_inputs.append(t)
clip_tensor_original = torch.stack(clip_inputs)
with torch.no_grad():
    orig_emb = clip_model.encode_image(clip_tensor_original)
    orig_emb = orig_emb / orig_emb.norm(dim=-1, keepdim=True)
print(f"  original CLIP embeddings: {orig_emb.shape}")

# ── 7. Interpolate, decode, measure CLIP similarity ─────────────────────────
print("\n[7] Interpolating, decoding, measuring CLIP similarity...")

results = {alpha: [] for alpha in ALPHAS}

for alpha in ALPHAS:
    print(f"\n  === α = {alpha} ===")
    t0 = time.time()
    
    # z_alpha = alpha * z_test + (1-alpha) * z_prototype
    z_alpha = alpha * z_test + (1 - alpha) * z_prototypes.to(z_test.dtype)
    
    # Decode in batches
    rec_embs = []
    n_batches = (len(z_alpha) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for b in tqdm(range(n_batches), desc=f"  α={alpha} decode"):
        z_batch = z_alpha[b*BATCH_SIZE:(b+1)*BATCH_SIZE].to(DEVICE)
        with torch.no_grad():
            # decode: AutoencoderKL decode expects latent in latent space (scale factor applied)
            # diffusers vae: decode returns decoded image in [0,1] or [-1,1]?
            rec = vae.decode(z_batch.float()).sample
            # rec: [B, 3, 512, 512] (float32)
        
        # Convert to CLIP input: resize back to 224x224, normalize
        rec_pil_images = []
        for rec_img in rec.float():
            # rec is in roughly [0,1] range (VAE decoder output)
            # Resize to 224 for CLIP
            rec_np = rec_img.permute(1,2,0).cpu().numpy()  # [H,W,3]
            rec_np = np.clip(rec_np, 0, 1)
            pil_img = Image.fromarray((rec_np * 255).astype(np.uint8))
            rec_pil_images.append(pil_img)
        
        # CLIP encode
        clip_batch = torch.stack([clip_preprocess(img).to(DEVICE) for img in rec_pil_images])
        with torch.no_grad():
            emb = clip_model.encode_image(clip_batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        rec_embs.append(emb.cpu())
    
    rec_embs = torch.cat(rec_embs, dim=0)  # [test_n, D]
    
    # Cosine similarity with original
    sims = (orig_emb * rec_embs).sum(dim=-1).numpy()  # [test_n]
    mean_sim = float(np.mean(sims))
    std_sim  = float(np.std(sims))
    
    results[alpha] = {"mean": mean_sim, "std": std_sim, "individual": sims.tolist()}
    print(f"  α={alpha}: mean CLIP sim = {mean_sim:.4f} ± {std_sim:.4f} ({time.time()-t0:.1f}s)")

# ── 8. Save results ──────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "hypothesis_c_interpolation_results.txt")
with open(out_path, "w") as f:
    f.write("="*70 + "\n")
    f.write("HYPOTHESIS C: Semantic Attribute Erasure via Latent Space Interpolation\n")
    f.write("="*70 + "\n\n")
    f.write(f"Setup:\n")
    f.write(f"  Training set (prototypes): {TRAIN_N} CIFAR-10 train images\n")
    f.write(f"  Test set (natural):        {TEST_N} CIFAR-10 test images\n")
    f.write(f"  VAE: stabilityai/sd-vae-ft-mse\n")
    f.write(f"  CLIP: ViT-B/32\n")
    f.write(f"  Image resize: 32x32 -> 512x512 (LANCZOS)\n")
    f.write(f"  Distance metric: cosine (L2 equivalent after normalization)\n")
    f.write(f"  Device: {DEVICE}\n\n")
    f.write(f"Results:\n")
    f.write(f"{'Alpha':>8} | {'Mean CLIP Sim':>14} | {'Std':>8} | {'N':>5}\n")
    f.write("-"*50 + "\n")
    for alpha in ALPHAS:
        r = results[alpha]
        f.write(f"{alpha:8.1f} | {r['mean']:14.4f} | {r['std']:8.4f} | {TEST_N:5}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("INTERPRETATION:\n")
    # Check trend: is mean_sim higher for lower alpha?
    low_alpha_mean = (results[0.0]['mean'] + results[0.2]['mean']) / 2
    high_alpha_mean = (results[0.8]['mean'] + results[1.0]['mean']) / 2
    if low_alpha_mean > high_alpha_mean:
        trend = "CONFIRMED: Lower alpha (moving toward prototype) → higher CLIP sim"
    else:
        trend = "NOT CONFIRMED: No clear increase as alpha decreases"
    f.write(f"  Low-alpha avg (0.0, 0.2): {low_alpha_mean:.4f}\n")
    f.write(f"  High-alpha avg (0.8, 1.0): {high_alpha_mean:.4f}\n")
    f.write(f"  Trend: {trend}\n")
    
    f.write("\nFull per-alpha data:\n")
    for alpha in ALPHAS:
        r = results[alpha]
        f.write(f"\n  α={alpha}: mean={r['mean']:.6f}, std={r['std']:.6f}\n")

print(f"\n[DONE] Results saved to {out_path}")
print("\n=== SUMMARY ===")
for alpha in ALPHAS:
    print(f"  α={alpha}: CLIP sim = {results[alpha]['mean']:.4f}")