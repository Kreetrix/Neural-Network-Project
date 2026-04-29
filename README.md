HERE IS THE LINK TO THE DATASET  https://www.kaggle.com/datasets/trungit/coco25k?resource=download

Mb we can try this article https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8/

AT THE MOMENT TWO VERSIONS GAN and RGB

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

The **Large-U-Net Model** uses the official **COCO 2017** dataset (`val2017` and `unlabeled2017` splits) and employs a hyper-fast multi-threaded preprocessing script to optimize training.

1. **Download:** Obtain the `val2017` and `unlabeled2017` images from the [official COCO website](https://cocodataset.org/#download).
2. **Initial Folder Structure:** Place the raw folders inside a `coco_data/` directory in the root of your project:
   ```text
   Neural-Network-Project/
   └── coco_data/
       ├── val2017/
       └── unlabeled2017/
   ```
3. **Hyper-Speed Resizing (`Large-U-Net.ipynb`):** The data pipeline automatically converts these images to RGB, resizes them to 128x128 using LANCZOS resampling, and compresses them. The processed images are saved to `coco_data_128/` for lightning-fast GPU loading.

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

---

## ⚖️ Model Evaluation & Testing

To prove the effectiveness of the GAN architecture, the `tester.ipynb` notebook pits the two models against each other.

1. **Load Checkpoints:** The script loads the saved weights from both the U-Net and the cGAN.
2. **Unseen Data:** It passes a batch of unseen, black-and-white test images through both models.
3. **Visual Comparison:** It plots a side-by-side grid showing:
    * **Input:** The Grayscale Image
    * **Baseline:** The U-Net's prediction (usually muted/safe colors)
    * **Advanced:** The cGAN's prediction (usually vibrant/risky colors)
    * **Ground Truth:** The original real photo