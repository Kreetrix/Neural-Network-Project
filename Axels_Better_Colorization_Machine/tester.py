import os
print("1. Imports starting...")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

print("2. Imports finished successfully.")

# --- 1. MODEL ARCHITECTURE ---
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
        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)
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

print("3. Model Architecture defined.")

# --- 2. SETUP CPU & LOAD WEIGHTS ---
device = torch.device("cpu") 
model = UNet().to(device)
print("4. Empty brain created on CPU.")

save_path = 'best_colorizer.pth'
if os.path.exists(save_path):
    print("5. Found the saved brain file. Attempting to load...")
    checkpoint = torch.load(save_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"✅ Loaded Brain (Score: {checkpoint['best_loss']:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded old format weights.")
else:
    print(f"❌ Error: {save_path} not found.")

model.eval()

# --- 3. INFERENCE TO FILE ---
def instant_colorize_to_file(image_path, output_path="colorized_output.jpg"):
    print(f"\n--- Starting to process {image_path} ---")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    print("6. Opening Image...")
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((128, 128)) 
    img_np = np.array(img_resized)
    
    print("7. Converting to LAB color space (This is a common crash spot!)...")
    img_lab = rgb2lab(img_np).astype(np.float32)
    img_lab = torch.tensor(img_lab).permute(2, 0, 1)
    
    print("8. Normalizing inputs...")
    L_channel = img_lab[[0], ...] / 50.0 - 1.0
    L_input = L_channel.unsqueeze(0).to(device)
    
    print("9. Pushing image through the AI...")
    with torch.no_grad():
        ab_prediction = model(L_input)
    
    print("10. AI finished! Converting back to RGB...")
    L_plot = L_input.squeeze(0).cpu()
    ab_plot = ab_prediction.squeeze(0).cpu()
    L_denorm = (L_plot + 1.0) * 50.0
    ab_denorm = ab_plot * 110.0
    
    lab = torch.cat([L_denorm, ab_denorm], dim=0).numpy().transpose((1, 2, 0))
    final_rgb = np.clip(lab2rgb(lab), 0, 1)
    
    print("11. Saving to disk...")
    final_img_uint8 = (final_rgb * 255).astype(np.uint8)
    final_image = Image.fromarray(final_img_uint8)
    final_image.save(output_path)
    print(f"🎉 Success! Check your folder for: {output_path}")

# --- 4. RUN ---
if __name__ == '__main__':
    instant_colorize_to_file("Amir.jpg", "Amir_colorized.jpg")