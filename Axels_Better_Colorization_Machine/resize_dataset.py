import os
from PIL import Image
import concurrent.futures

# The folders we want to shrink
input_folders = ['./coco_data/val2017', './coco_data/unlabeled2017']
# Where we will save the tiny images
output_base = './coco_data_128'

def process_image(img_path, output_path):
    try:
        # If we already resized it, skip it!
        if os.path.exists(output_path): return 
        
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=90)
    except Exception as e:
        print(f"Error on {img_path}: {e}")

# Create the new folders if they don't exist
for folder in input_folders:
    new_folder = folder.replace('./coco_data', output_base)
    os.makedirs(new_folder, exist_ok=True)

# Gather all files
tasks = []
for folder in input_folders:
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
        for f in files:
            input_path = os.path.join(folder, f)
            output_path = os.path.join(folder.replace('./coco_data', output_base), f)
            tasks.append((input_path, output_path))

print(f"Found {len(tasks)} images. Starting hyper-speed resizing...")

# This uses a safe Windows thread pool to resize the images FAST
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_image, inp, out) for inp, out in tasks]
    for i, _ in enumerate(concurrent.futures.as_completed(futures)):
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1} / {len(tasks)} images...")

print("✅ Preprocessing Complete! Your dataset is now tiny and lightning fast.")