import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in range(10):
        label_folder = os.path.join(folder, str(label))
        for filename in os.listdir(label_folder):
            img = cv2.imread(os.path.join(label_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize to for HOG
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

def extract_hog_features(images):
    hog_features = []
    for image in images:
        features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys',
                                  visualize=True)
        hog_features.append(features)
    return np.array(hog_features)

# Load and preprocess images
X, y = load_images_from_folder('gesture language dataset')

# Extract HOG features from the images
X_hog = extract_hog_features(X)

# Create a DataFrame with the HOG features and their labels
data_hog = pd.DataFrame(X_hog)
data_hog['label'] = y

# Save the DataFrame to a CSV file
data_hog.to_csv('sign_language_digits_hog.csv', index=False)

# Load the data from CSV to verify
loaded_data_hog = pd.read_csv('sign_language_digits_hog.csv')

# Separate features and labels
X_loaded_hog = loaded_data_hog.drop('label', axis=1).values
y_loaded_hog = loaded_data_hog['label'].values

# use PCA to reduce number of features
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X_loaded_hog)

# Create a DataFrame with the reduced features and their labels
data_reduced = pd.DataFrame(X_reduced)
data_reduced['label'] = y_loaded_hog

# Save the DataFrame to a CSV file
data_reduced.to_csv('sign_language_digits_reduced.csv', index=False)

# Load the data from CSV to verify
loaded_data_reduced = pd.read_csv('sign_language_digits_reduced.csv')

# Separate features and labels
X_loaded_reduced = loaded_data_reduced.drop('label', axis=1).values
y_loaded_reduced = loaded_data_reduced['label'].values

# Verify the data
print(loaded_data_reduced.head())


