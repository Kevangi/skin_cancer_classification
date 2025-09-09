import cv2
import numpy as np
import os

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

def hair_removal(img):
    return dull_razor(img)

def process_batch(input_folder, output_folder, batch_size=20):
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))

    total_images = len(image_paths)
    print(f"Total images to process: {total_images}")

    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch_paths)} images")

        for img_path in batch_paths:
            rel_path = os.path.relpath(img_path, input_folder)
            output_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to load {img_path}. Skipping.")
                continue

            cleaned_img = hair_removal(img)
            cv2.imwrite(output_path, cleaned_img)

    print("✅ Hair removal completed for all images.")

# Example usage
if __name__ == "__main__":
    process_batch("dataset/train", "dataset/train_nohair", batch_size=20)
    process_batch("dataset/val", "dataset/val_nohair", batch_size=20)
