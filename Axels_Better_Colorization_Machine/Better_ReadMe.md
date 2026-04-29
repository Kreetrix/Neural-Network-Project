## 💾 Dataset Requirements

To train or test this model, you need the **COCO 2017 Dataset**. 

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
   ├── colorizer_training.ipynb
   ├── tester.py
   ├── resize_dataset.py
   └── requirements.txt

3. **install requirments**
   `pip install -r requirements.txt`
   `python resize_dataset.py`