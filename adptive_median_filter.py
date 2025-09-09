import cv2
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool

def adaptive_median_filter(img, max_filter_size=5):
    """
    Apply adaptive median filter to remove noise.
    """
    height, width = img.shape[:2]
    output_img = np.copy(img)

    for i in range(height):
        for j in range(width):
            filter_size = 3  # Minimum filter size
            while filter_size <= max_filter_size:
                # Extract the neighborhood
                half_size = filter_size // 2
                x_min, x_max = max(i - half_size, 0), min(i + half_size + 1, height)
                y_min, y_max = max(j - half_size, 0), min(j + half_size + 1, width)
                neighborhood = img[x_min:x_max, y_min:y_max]
                
                # Apply median filter
                median = np.median(neighborhood)
                
                # Apply decision rule
                if np.all(median != img[i, j]):
                    output_img[i, j] = median
                    break
                filter_size += 2  # Increase filter size

    return output_img

def process_image(image_path):
    """Process a single image file."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return adaptive_median_filter(img)

def process_images_in_directory(directory_path, output_directory):
    """Process all images in a directory in parallel."""
    # Create a pool of workers (CPU cores) for parallel processing
    image_paths = []
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                if image_name.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(folder_path, image_name))

    # Create a pool of workers and process the images in parallel
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        processed_images = pool.map(process_image, image_paths)

    # Save the processed images
    for idx, image_path in enumerate(image_paths):
        folder_name = os.path.basename(os.path.dirname(image_path))
        output_folder = os.path.join(output_directory, folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_image_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, processed_images[idx])

if __name__ == "__main__":
    input_dir = "dataset/train_nohair"  # Input directory with train_nohair -> bcc, mel, etc.
    output_dir = "dataset/filtered_images"  # Output directory where processed images will be saved
    process_images_in_directory(input_dir, output_dir)
