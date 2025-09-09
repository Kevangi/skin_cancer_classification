import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

def dull_razor(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Morphological closing to highlight hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint to remove hair
    result = cv2.inpaint(img, hair_mask, 1, cv2.INPAINT_TELEA)
    return result

def compare_images(original_img_path, cleaned_img):
    # Load the original image
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Warning: Unable to load {original_img_path}. Skipping.")
        return

    # Convert BGR to RGB for proper plotting
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    cleaned_img_rgb = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)

    # Plot side-by-side comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_img_rgb)
    plt.title("Hair Removed")
    plt.axis('off')

    plt.show()

def process_and_compare_one(input_folder, output_folder, batch_size=20):
    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))

    total_images = len(image_paths)
    print(f"Total images to process: {total_images}")

    # Select a random image path
    random_image_path = random.choice(image_paths)
    print(f"Processing random image: {random_image_path}")

    # Create output path
    rel_path = os.path.relpath(random_image_path, input_folder)
    output_path = os.path.join(output_folder, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read and process the random image
    img = cv2.imread(random_image_path)
    if img is None:
        print(f"Warning: Unable to load {random_image_path}. Skipping.")
        return

    cleaned_img = dull_razor(img)
    cv2.imwrite(output_path, cleaned_img)

    # Compare the original image with the hair-removed version
    compare_images(random_image_path, cleaned_img)

    print("✅ Hair removal and comparison completed for the selected image.")

# Example usage
if __name__ == "__main__":
    process_and_compare_one("dataset/train", "dataset/train_nohair", batch_size=20)
    process_and_compare_one("dataset/val", "dataset/val_nohair", batch_size=20)
