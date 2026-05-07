"""
compare_models.py
Runs all available model checkpoints on the same test images and saves
side-by-side comparison grids to report_images/.
"""

import os, math, warnings, random
import numpy as np
from PIL import Image
from skimage import color as skcolor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

os.makedirs('report_images', exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cpu')
print(f'Device: {device}')

# ── Shared helpers ────────────────────────────────────────────────────────
def lab_to_rgb(L_t, ab_t):
    L  = (L_t.squeeze().numpy() + 1.0) * 50.0
    ab = ab_t.permute(1, 2, 0).numpy() * 110.0
    lab = np.concatenate([L[:, :, None], ab], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return skcolor.lab2rgb(lab.astype(np.float64)).clip(0, 1)

def load_image(path, size=128):
    img = Image.open(path).convert('RGB').resize((size, size), Image.BICUBIC)
    rgb_np = np.array(img, dtype=np.float32) / 255.0
    lab = skcolor.rgb2lab(rgb_np).astype(np.float32)
    L_t  = torch.from_numpy(lab[:, :, 0:1].transpose(2, 0, 1)) / 50.0 - 1.0
    ab_t = torch.from_numpy(lab[:, :, 1:3].transpose(2, 0, 1)) / 110.0
    return L_t, ab_t

def mse_to_psnr(mse):
    return 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')

# ── Architecture definitions ──────────────────────────────────────────────

# --- Large U-Net (Axels_Better_Colorization_Machine) ---
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class LargeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 64);  self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128); self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        x1 = self.down1(x); x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2)); bn = self.bottleneck(self.pool3(x3))
        u1 = self.up_conv1(torch.cat([self.up1(bn), x3], 1))
        u2 = self.up_conv2(torch.cat([self.up2(u1), x2], 1))
        u3 = self.up_conv3(torch.cat([self.up3(u2), x1], 1))
        return torch.tanh(self.out_conv(u3))

# --- GAN Generator (colorization_gan / colorization_final) ---
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

class GANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_block(1, 64);  self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256); self.pool = nn.MaxPool2d(2)
        self.mid  = conv_block(256, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, 2); self.dec3 = conv_block(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, 2); self.dec2 = conv_block(256, 128)
        self.up1  = nn.ConvTranspose2d(128,  64, 2, 2); self.dec1 = conv_block(128,  64)
        self.out  = nn.Conv2d(64, 2, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        m  = self.mid(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.tanh(self.out(d1))

# ── Model registry ────────────────────────────────────────────────────────
MODELS = [
    {
        'name': 'Large U-Net\n(128k images, L1)',
        'ckpt': 'Axels_Better_Colorization_Machine/best_colorizer.pth',
        'arch': LargeUNet,
        'key':  'model_state',
    },
    {
        'name': 'Single-Stage cGAN\n(2k images, L1+Adv)',
        'ckpt': 'generator_lab.pth',
        'arch': GANGenerator,
        'key':  'model_state',
    },
    {
        'name': 'Two-Stage cGAN\n(8k images, L1+Adv+VGG)',
        'ckpt': 'generator_final.pth',
        'arch': GANGenerator,
        'key':  'model_state',
    },
]

def try_load(entry):
    path = entry['ckpt']
    if not os.path.exists(path):
        print(f'  [SKIP] {path} not found')
        return None
    model = entry['arch']().to(device)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    key   = entry['key']
    state = ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f'  [OK]   {path}')
    return model

print('\nLoading models...')
loaded = [(m['name'], try_load(m)) for m in MODELS]
loaded = [(name, mdl) for name, mdl in loaded if mdl is not None]
print(f'{len(loaded)} model(s) loaded.\n')

# ── Pick test images ──────────────────────────────────────────────────────
IMG_DIR = 'data/images'
all_imgs = sorted([f for f in os.listdir(IMG_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
random.seed(7)
test_imgs = random.sample(all_imgs, min(6, len(all_imgs)))

# ── Per-image comparison grid ─────────────────────────────────────────────
n_models  = len(loaded)
n_cols    = 2 + n_models          # grayscale + ground truth + one per model
n_rows    = len(test_imgs)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3 * n_cols, 3 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

col_titles = ['Grayscale\n(input)', 'Ground Truth'] + \
             [name for name, _ in loaded]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=9, fontweight='bold')

psnr_table = {name: [] for name, _ in loaded}

for row, fname in enumerate(test_imgs):
    path = os.path.join(IMG_DIR, fname)
    L_t, ab_t = load_image(path)

    # grayscale
    gray = (L_t.squeeze().numpy() + 1.0) / 2.0
    axes[row, 0].imshow(gray, cmap='gray')
    axes[row, 0].set_ylabel(fname[:18], fontsize=7)

    # ground truth
    axes[row, 1].imshow(lab_to_rgb(L_t, ab_t))

    for col, (name, model) in enumerate(loaded, start=2):
        with torch.no_grad():
            pred_ab = model(L_t.unsqueeze(0).to(device)).squeeze(0).cpu()
        pred_rgb = lab_to_rgb(L_t, pred_ab)
        axes[row, col].imshow(pred_rgb)
        mse  = float(((pred_ab - ab_t) ** 2).mean())
        psnr = mse_to_psnr(mse)
        psnr_table[name].append(psnr)
        axes[row, col].set_xlabel(f'PSNR {psnr:.1f} dB', fontsize=7)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Model comparison — same test images', fontsize=13, y=1.01)
plt.tight_layout()
out = 'report_images/comparison_grid.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')

# ── PSNR bar chart ────────────────────────────────────────────────────────
if psnr_table:
    names = list(psnr_table.keys())
    avgs  = [sum(v) / len(v) for v in psnr_table.values()]
    colors = ['#4C72B0', '#DD8452', '#55A868'][:len(names)]

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(names)), 4))
    bars = ax.bar([n.replace('\n', ' ') for n in names], avgs,
                  color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f} dB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Average PSNR (dB)')
    ax.set_title('Average PSNR on test images (higher = better)')
    ax.set_ylim(0, max(avgs) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out2 = 'report_images/psnr_comparison.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out2}')

    print('\nAverage PSNR per model:')
    for name, avg in zip(names, avgs):
        print(f'  {name.replace(chr(10), " "):45s}  {avg:.2f} dB')

# ── Per-model best/worst strip ────────────────────────────────────────────
for name, model in loaded:
    scores = []
    all_files = all_imgs[:50]
    for fname in all_files:
        L_t, ab_t = load_image(os.path.join(IMG_DIR, fname))
        with torch.no_grad():
            pred_ab = model(L_t.unsqueeze(0).to(device)).squeeze(0).cpu()
        mse = float(((pred_ab - ab_t) ** 2).mean())
        scores.append((mse_to_psnr(mse), fname))
    scores.sort(key=lambda x: x[0])
    worst3 = scores[:3]
    best3  = scores[-3:][::-1]

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for col, (psnr, fname) in enumerate(best3):
        L_t, ab_t = load_image(os.path.join(IMG_DIR, fname))
        with torch.no_grad():
            pred_ab = model(L_t.unsqueeze(0).to(device)).squeeze(0).cpu()
        axes[0, col*2].imshow(lab_to_rgb(L_t, ab_t))
        axes[0, col*2].set_title(f'GT  PSNR={psnr:.1f}dB', fontsize=8)
        axes[0, col*2+1].imshow(lab_to_rgb(L_t, pred_ab))
        axes[0, col*2+1].set_title('Predicted', fontsize=8)
        for ax in [axes[0, col*2], axes[0, col*2+1]]: ax.axis('off')

    for col, (psnr, fname) in enumerate(worst3):
        L_t, ab_t = load_image(os.path.join(IMG_DIR, fname))
        with torch.no_grad():
            pred_ab = model(L_t.unsqueeze(0).to(device)).squeeze(0).cpu()
        axes[1, col*2].imshow(lab_to_rgb(L_t, ab_t))
        axes[1, col*2].set_title(f'GT  PSNR={psnr:.1f}dB', fontsize=8)
        axes[1, col*2+1].imshow(lab_to_rgb(L_t, pred_ab))
        axes[1, col*2+1].set_title('Predicted', fontsize=8)
        for ax in [axes[1, col*2], axes[1, col*2+1]]: ax.axis('off')

    axes[0, 0].set_ylabel('Best results', fontsize=10, rotation=90, labelpad=5)
    axes[1, 0].set_ylabel('Worst results', fontsize=10, rotation=90, labelpad=5)
    safe = name.split('\n')[0].replace(' ', '_').replace('/', '_')
    plt.suptitle(f'{name.replace(chr(10), " ")} — best & worst', fontsize=12)
    plt.tight_layout()
    out3 = f'report_images/best_worst_{safe}.png'
    plt.savefig(out3, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out3}')

