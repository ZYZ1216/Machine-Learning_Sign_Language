import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

# Input images with labels
def load_images_and_labels(dataset_path):
    images = []
    labels = []
    for digit in range(10):
        folder_path = os.path.join(dataset_path, str(digit))
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # convert to grayscale
            img = cv2.resize(img, (32, 32)) # resize to 32x32 pixels for every image
            images.append(img)
            labels.append(digit)
    return images, labels

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)

# Save features and labels after using HOG to CSV
def save_to_csv(features, labels, filename):
    data = np.column_stack((features, labels))
    columns = [f'feature_{i}' for i in range(features.shape[1])] + ['label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

# Load and process the dataset
dataset_path = 'Sign_Language_Digits_Dataset'
images, labels = load_images_and_labels(dataset_path)
hog_features = extract_hog_features(images)
labels = np.array(labels)

# Save to CSV
csv_filename = 'gesture_language_digits_hog_features.csv'
save_to_csv(hog_features, labels, csv_filename)
# 323 features extracted by using HOG
print(f"Features and labels saved to {csv_filename}")
