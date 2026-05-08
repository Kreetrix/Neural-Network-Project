Here is link to the dataset  https://www.kaggle.com/datasets/trungit/coco25k?resource=download

We used this article as source of inspiration : https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8/

##  Setup & Installation Guide

This project is optimized for **AMD GPUs** via Microsoft DirectML, but supports NVIDIA and CPU-only setups. To avoid DLL errors and environment "leaks," follow these specific steps.

### 1. Recommended Environment (Conda)
Because this project requires specific C++ runtimes for AMD hardware acceleration, **Anaconda/Conda** is the most stable choice.

```bash
# Create the environment
conda create -n colorize python=3.10
conda activate colorize

# Install the verified "Stable Triad" for AMD GPUs
pip install torch==2.4.1 torchvision==0.19.1 torch-directml==0.2.5.dev240914 numpy==2.0.1

# Install project dependencies
pip install -r requirements.txt
```

### 2. VS Code Integration
1. Open the project folder in VS Code.
2. Press `Ctrl+Shift+P` -> **Python: Select Interpreter**.
3. Choose the `colorize` environment.
4. Open your notebook and ensure the kernel in the top-right matches this environment.

---

##  Dataset & Preprocessing

1. **Download the Data:**
   Go to the official [COCO Dataset website](https://cocodataset.org/#download) and download the following files:
   * `2017 Val images` (1GB)
   * `2017 Unlabeled images` (19GB) - *Optional, but recommended for full training.*

2. **Folder Structure:**
   Extract the downloaded `.zip` files and place them inside a `coco_data` folder in the root of this project. Your directory should look exactly like this:

   ```text
   your_project_folder/
   │
   ├── coco_data/
   │   ├── val2017/          <-- Put the 5,000 val images here
   │   └── unlabeled2017/    <-- Put the 123,000 unlabeled images here
   │
   ├── Large-U-Net.ipynb
   ├── tester.py
   ├── resize_dataset.py
   └── requirements.txt

---

##  The Models

### 1. The U-Net (Generator)
The core of our colorization engine is a U-Net. It uses an **Encoder-Decoder** structure with skip-connections to preserve high-resolution spatial details from the original grayscale image.
- **Input:** $L$ channel (Lightness) from the Lab color space.
- **Output:** $a$ and $b$ channels (Chrominance).

### 2. The Conditional GAN (cGAN)
While U-Nets often produce "sepia" or brownish results because they minimize average error, the GAN introduces a **Discriminator**. 
- **The Generator** tries to create a realistic color image.
- **The Discriminator** acts as a critic, learning to distinguish between "fake" colorized images and "real" ground-truth photos.
- **Result:** This adversarial battle forces the model to produce more vibrant and diverse colors.
### 3. The Pix2Pix Architecture (Our Custom GAN)
Pix2Pix is the specific architecture we built by combining the U-Net Generator and the cGAN framework. Rather than a standard critic, it introduces a specialized **PatchGAN** and a dual-loss mathematical anchor.
- **The PatchGAN Critic:** Instead of judging the entire image at once, the Discriminator breaks the image down into smaller patches (e.g., 70x70 pixels) to judge local texture and realism, making it highly efficient.
- **The L1 Anchor:** Alongside the adversarial GAN game, the Generator is heavily penalized for straying too far from the exact colors of the ground-truth reference photo (L1 Loss).
- **Result:** The ultimate balance. The GAN forces vibrant, realistic textures, while the heavy L1 anchor prevents the AI from hallucinating chaotic neon colors.
---

##  Model Evaluation & Testing

To prove the effectiveness of the GAN architecture, the `tester.ipynb` notebook pits the two models against each other.

1. **Load Checkpoints:** The script loads the saved weights from both the U-Net and the cGAN.
2. **Unseen Data:** It passes a batch of unseen, black-and-white test images through both models.
3. **Visual Comparison:** It plots a side-by-side grid showing:
    * **Input:** The Grayscale Image
    * **Baseline:** The U-Net's prediction (usually muted/safe colors)
    * **Advanced:** The cGAN's prediction (usually vibrant/risky colors)
    * **Ground Truth:** The original real photo
