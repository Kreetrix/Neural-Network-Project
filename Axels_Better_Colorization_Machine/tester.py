import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

print("1. Imports finished successfully.")

# ==========================================
# ARCHITECTURE 1: STANDARD U-NET (Early Stopping)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 64); self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128); self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.up_conv3 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        x1 = self.down1(x); p1 = self.pool1(x1)
        x2 = self.down2(p1); p2 = self.pool2(x2)
        x3 = self.down3(p2); p3 = self.pool3(x3)
        bn = self.bottleneck(p3)
        u1 = torch.cat([self.up1(bn), x3], dim=1); u1 = self.up_conv1(u1)
        u2 = torch.cat([self.up2(u1), x2], dim=1); u2 = self.up_conv2(u2)
        u3 = torch.cat([self.up3(u2), x1], dim=1); u3 = self.up_conv3(u3)
        return torch.tanh(self.out_conv(u3))


# ==========================================
# ARCHITECTURE 2: GAN GENERATOR
# ==========================================
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class GAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_block(1,   64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.mid  = conv_block(256, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, 2); self.dec3 = conv_block(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, 2); self.dec2 = conv_block(256, 128)
        self.up1  = nn.ConvTranspose2d(128,  64, 2, 2); self.dec1 = conv_block(128,  64)
        self.out  = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m  = self.mid(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.tanh(self.out(d1))


# ==========================================
# SETUP AND LOAD MODELS
# ==========================================
device = torch.device("cpu")
print("2. Initializing models on CPU...")

unet_model = UNet().to(device)
gan_model = GAN_Generator().to(device)
unet_loss = "N/A"

# Load U-Net
unet_path = 'best_colorizer.pth'
if os.path.exists(unet_path):
    checkpoint = torch.load(unet_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        unet_model.load_state_dict(checkpoint['model_state'])
        unet_loss = f"{checkpoint['best_loss']:.4f}"
    else:
        unet_model.load_state_dict(checkpoint)
    print(f"✅ U-Net loaded. Best Val Loss: {unet_loss}")
else:
    print(f"❌ Error: {unet_path} not found.")

# Load GAN
gan_path = 'generator_lab.pth'
gan_loss = "N/A"

if os.path.exists(gan_path):
    checkpoint = torch.load(gan_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        gan_model.load_state_dict(checkpoint['model_state'])
        if 'val_loss' in checkpoint:
            gan_loss = f"{checkpoint['val_loss']:.4f}"
        elif 'best_loss' in checkpoint:
            gan_loss = f"{checkpoint['best_loss']:.4f}"
        print(f"✅ GAN Generator loaded. Val Loss: {gan_loss}")
    else:
        gan_model.load_state_dict(checkpoint)
        print("✅ GAN Generator loaded. (Old save format - No Val Loss)")
else:
    print(f"❌ Error: {gan_path} not found.")

unet_model.eval()
gan_model.eval()

# ==========================================
# COMPARISON INFERENCE FUNCTION
# ==========================================
def compare_models(image_path, output_path="model_comparison.jpg"):
    print(f"\n--- Processing {image_path} ---")
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    # Prepare Image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((128, 128))
    img_np = np.array(img_resized)
    
    img_lab = rgb2lab(img_np).astype(np.float32)
    img_lab_tensor = torch.tensor(img_lab).permute(2, 0, 1)
    
    L_channel = img_lab_tensor[[0], ...] / 50.0 - 1.0
    L_input = L_channel.unsqueeze(0).to(device)

    # Run Inference
    with torch.no_grad():
        ab_unet = unet_model(L_input)
        ab_gan = gan_model(L_input)

    # Helper function to convert tensors to RGB
    def to_rgb(L_tensor, ab_tensor):
        L = (L_tensor.squeeze(0).cpu() + 1.0) * 50.0
        ab = ab_tensor.squeeze(0).cpu() * 110.0
        lab = torch.cat([L, ab], dim=0).numpy().transpose((1, 2, 0))
        return np.clip(lab2rgb(lab), 0, 1)

    print("🎨 Reconstructing colors...")
    rgb_unet = to_rgb(L_input, ab_unet)
    rgb_gan = to_rgb(L_input, ab_gan)
    
    # Create Grayscale version for reference
    L_only_lab = torch.cat([
        (L_input.squeeze(0).cpu() + 1.0) * 50.0, 
        torch.zeros(2, 128, 128)
    ], dim=0).numpy().transpose((1, 2, 0))
    rgb_gray = np.clip(lab2rgb(L_only_lab), 0, 1)

    # ==========================================
    # VISUALIZATION
    # ==========================================
    print("📸 Generating comparison plot...")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5)) 
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original (Ground Truth)")
    axes[0].axis('off')

    axes[1].imshow(rgb_gray)
    axes[1].set_title("Input (Grayscale)")
    axes[1].axis('off')

    axes[2].imshow(rgb_unet)
    axes[2].set_title(f"Standard U-Net\nVal Loss: {unet_loss}")
    axes[2].axis('off')

    axes[3].imshow(rgb_gan)
    axes[3].set_title(f"GAN Generator\nVal Loss: {gan_loss}") 
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"🎉 Success! Comparison saved to: {output_path}")
    plt.show() 

if __name__ == '__main__':
    compare_models("Amir.jpg", "Amir_comparison.jpg")