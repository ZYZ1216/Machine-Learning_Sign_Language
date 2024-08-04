import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

## Load and preprocess the imagesfhfjfjhfyjfhf  1111
def load_images_from_folder(new_path):
    images = []
    labels = []
    for filename in os.listdir(new_path):
        img = cv2.imread(os.path.join(new_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize every image to a consistent size
            label = int(filename.split('_')[0])  # Assuming filename format is 'label_xxx.jpg'
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Extract HOG features from the images
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        hog_features.append(features)
    return np.array(hog_features)

# Load dataset
folder_path = 'path_to_your_dataset_folder'
X, y = load_images_from_folder(folder_path)

# Extract HOG features
X_hog = extract_hog_features(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize a sample prediction


sample_index = 0  # Change this index to visualize different samples
sample_image = X[sample_index]
plt.imshow(sample_image, cmap='gray')
plt.title(f"True label: {y_test[sample_index]}, Predicted label: {y_pred[sample_index]}")
plt.show()
