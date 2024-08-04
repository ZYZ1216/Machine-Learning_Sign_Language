import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog
from skimage import color

# Directory containing the augmented dataset
dataset_dir = 'Augmented_Sign_Language_Digits_Dataset'
# Desired image size (width, height)
image_size = (32, 32)

# Function to resize images to a consistent size
def resize_image(image, size):
    return image.resize(size, Image.LANCZOS)

# Function to extract pixel values from an image
def extract_pixel_values(image):
    gray_image = color.rgb2gray(np.array(image))
    return np.array(gray_image).flatten()

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Convert image to grayscale
    gray_image = color.rgb2gray(np.array(image))
    hog_features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features
# Function to process images and extract features
def process_images(input_dir, feature_type='pixel'):
    features = []
    labels = []
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            try:
                image = Image.open(image_path).convert('RGB')
                image = resize_image(image, image_size)  # Resize image to consistent size
                if feature_type == 'pixel':
                    image_features = extract_pixel_values(image)
                elif feature_type == 'hog':
                    image_features = extract_hog_features(image)
                else:
                    raise ValueError("Invalid feature type. Use 'pixel' or 'hog'.")
                features.append(image_features)
                labels.append(label)
            except UnidentifiedImageError:
                print(f"Skipping unreadable image: {image_path}")
                continue
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
    features_array = np.array(features)
    print(f"Processed {len(features)} images for feature type '{feature_type}'. Feature array shape: {features_array.shape}")
    return features_array, np.array(labels)

# Function to save features to CSV
def save_features_to_csv(features, labels, output_csv, feature_type='pixel'):
    if features.size == 0:
        print("No features to save.")
        return
    if feature_type == 'pixel':
        columns = [f'pixel_{i+1}' for i in range(features.shape[1])]
    elif feature_type == 'hog':
        columns = [f'feature_{i+1}' for i in range(features.shape[1])]
    else:
        raise ValueError("Invalid feature type. Use 'pixel' or 'hog'.")
    df = pd.DataFrame(features, columns=columns)
    df['label'] = labels
    df.to_csv(output_csv, index=False)

# Extract and save pixel features
pixel_features, pixel_labels = process_images(dataset_dir, feature_type='pixel')
save_features_to_csv(pixel_features, pixel_labels, 'pixel_features.csv', feature_type='pixel')

# Extract and save HOG features
hog_features, hog_labels = process_images(dataset_dir, feature_type='hog')
save_features_to_csv(hog_features, hog_labels, 'hog_features.csv', feature_type='hog')

print("Feature extraction and saving to CSV completed.")
