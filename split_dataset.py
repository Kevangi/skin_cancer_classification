import os
import random
import shutil
from tqdm import tqdm
import pandas as pd

# Paths
metadata_path = 'dataset/HAM10000_metadata.csv'
source_dir = 'dataset/all_images'
train_dir = 'dataset/train'
val_dir = 'dataset/val'
csv_dir = 'dataset'  # Where train.csv and val.csv will be saved

# Load metadata
df = pd.read_csv(metadata_path)
df['file_name'] = df['image_id'] + '.jpg'
label_map = dict(zip(df['file_name'], df['dx']))

# List of all image filenames that exist and are labeled
all_images = [f for f in os.listdir(source_dir) if f in label_map]
random.shuffle(all_images)

# Split
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Containers for CSV entries
train_csv_entries = []
val_csv_entries = []

# Helper to copy images and store CSV rows
def copy_and_log(image_list, subset_dir, entries_list, desc):
    for img_name in tqdm(image_list, desc=desc):
        label = label_map[img_name]
        src_path = os.path.join(source_dir, img_name)
        dest_dir = os.path.join(subset_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, img_name)
        shutil.copy(src_path, dest_path)

        relative_path = os.path.relpath(dest_path, start=csv_dir)  # for cleaner CSV
        entries_list.append((relative_path.replace("\\", "/"), label))

# Process train and val
copy_and_log(train_images, train_dir, train_csv_entries, "Train Images")
copy_and_log(val_images, val_dir, val_csv_entries, "Val Images")

# Save CSVs
pd.DataFrame(train_csv_entries, columns=["image_path", "label"]).to_csv(os.path.join(csv_dir, "train.csv"), index=False)
pd.DataFrame(val_csv_entries, columns=["image_path", "label"]).to_csv(os.path.join(csv_dir, "val.csv"), index=False)

print(f"\n✅ All done!")
print(f"  → {len(train_images)} images copied to '{train_dir}/[label]/' and listed in 'train.csv'")
print(f"  → {len(val_images)} images copied to '{val_dir}/[label]/' and listed in 'val.csv'")
