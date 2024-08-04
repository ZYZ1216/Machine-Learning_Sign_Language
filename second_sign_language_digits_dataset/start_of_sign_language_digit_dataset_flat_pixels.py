import os
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# import images from dataset
def load_images_and_labels_to_data(dataset_path):
    images = []
    labels = []
    for digit in range(10):
        folder_path = os.path.join(dataset_path, str(digit))
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # convert to grayscale
            img = cv2.resize(img, (32, 32)) # rescale to the same size for every image
            images.append(img.flatten()) # flatten the image to each pixel as one feature
            labels.append(digit)
    return np.array(images), np.array(labels)

# write in data source
dataset_path = 'Sign_Language_Digits_Dataset'
images, labels = load_images_and_labels_to_data(dataset_path)

# Combine images and labels into a single DataFrame
data = np.column_stack((images, labels))

# Create headers for the image pixel columns and the label column
num_pixels = images.shape[1]
headers = [f'pixel_{i}' for i in range(num_pixels)] + ['label']

# Convert data to DataFrame with headers
df = pd.DataFrame(data, columns=headers)

# Save DataFrame to a CSV file
csv_file_path = 'images_and_labels.csv'
df.to_csv(csv_file_path, index=False)
# 1023 pixels as features
print(f'Data saved to {csv_file_path}')


# Split the data for training
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# add a new function to save model results into a txt file
def save_results_to_file(filename, model_name, accuracy, report):
    with open(filename, 'a') as f:
        f.write(f"{model_name} Classifier\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write("\n" + "="*80 + "\n\n")

# Train and evaluate three machine learning models with plain pixel features: KNN, Naive Bayesian Classifier, and Random Forest
# 1. KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, knn_predictions)
report_knn = classification_report(y_test, knn_predictions)
print("KNN Accuracy:", accuracy_score(y_test, knn_predictions))
save_results_to_file('model_results_plain_pixel.txt', 'KNN', accuracy_knn, report_knn)


# 2. Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predications = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, nb_predications)
report_nb = classification_report(y_test, nb_predications)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predications))
save_results_to_file('model_results_plain_pixel.txt', 'Naive Bayes', accuracy_nb, report_nb)

# 3. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_predications = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, rf_predications)
report_rf = classification_report(y_test, rf_predications)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predications))
save_results_to_file('model_results_plain_pixel.txt', 'Random Forest', accuracy_rf, report_rf)

# Apply PCA to reduce dimensions and select 50 most important features
pca = PCA(n_components=50)
images_reduced_pca = pca.fit_transform(images)

# Combine reduced features with labels
data_reduced_pca = np.column_stack((images_reduced_pca, labels))
df_reduced_pca = pd.DataFrame(data_reduced_pca)

# Save to a new CSV file for reduced features with PCA
csv_file_path_reduced_pca = 'images_and_labels_reduced_pca.csv'
df_reduced_pca.to_csv(csv_file_path_reduced_pca, index=False)
print(f'Reduced data saved to {csv_file_path_reduced_pca}')


# Apply SelectKBest to select top 50 features
selector = SelectKBest(chi2, k=50)
images_reduced_skb = selector.fit_transform(images, labels)
data_reduced_skb = np.column_stack((images_reduced_skb, labels))

# convert stack into a DataFrame
df_reduced_skb = pd.DataFrame(data_reduced_skb)

# Save to a new CSV with name using SelectKBest
csv_file_path_reduced_skb = 'images_and_labels_reduced_SelectKBest.csv'
df_reduced_skb.to_csv(csv_file_path_reduced_skb, index=False)
print(f'Reduced data saved to {csv_file_path_reduced_skb}')


# Train a Random Forest model with 100 estimators
rf = RandomForestClassifier(n_estimators=100)
rf.fit(images, labels)

# reduce number of features using Random Forest model by ranking the features' importance
importances = rf.feature_importances_

# select k features based on importance of features
k = 50
indices_rf = np.argsort(importances)[-k:]

# Select the top 50 features
images_reduced_rf = images[:, indices_rf]

# Combine reduced images with labels
data_reduced_rf = np.column_stack((images_reduced_rf, labels))

# Create a DataFrame with headers for reduced features and labels
headers = [f'pixel_{i}' for i in indices_rf] + ['label']
df_reduced_rf = pd.DataFrame(data_reduced_rf, columns=headers)

# Save to a new CSV with reduced features
csv_file_path_reduced_rf = 'images_and_labels_reduced_rf.csv'
df_reduced_rf.to_csv(csv_file_path_reduced_rf, index=False)
print(f'Reduced data saved to {csv_file_path_reduced_rf}') 
