import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Load the CSV files
pixel_features_df = pd.read_csv('pixel_features.csv')
hog_features_df = pd.read_csv('hog_features.csv')

# Ensure the label column is the last column
def move_label_to_end(df):
    columns = df.columns.tolist()
    columns.append(columns.pop(columns.index('label')))
    return df[columns]

pixel_features_df = move_label_to_end(pixel_features_df)
hog_features_df = move_label_to_end(hog_features_df)

# Separate features and labels
X_pixel = pixel_features_df.drop(columns=['label'])
y_pixel = pixel_features_df['label']
X_hog = hog_features_df.drop(columns=['label'])
y_hog = hog_features_df['label']

# Encode the labels
le = LabelEncoder()
y_pixel_encoded = le.fit_transform(y_pixel)
y_hog_encoded = le.fit_transform(y_hog)

# Split the data into training (70%) and testing (30%)
X_pixel_train, X_pixel_test, y_pixel_train, y_pixel_test = train_test_split(X_pixel, y_pixel_encoded, test_size=0.2, random_state=42)
X_hog_train, X_hog_test, y_hog_train, y_hog_test = train_test_split(X_hog, y_hog_encoded, test_size=0.2, random_state=42)

# mRMR Feature Selection
def compute_redundancy(feature, best_feature, X):
    return mutual_info_regression(X[[best_feature]].copy(), X[feature].copy())[0]


def mrmr_feature_selection(X, y, n_features, sample_size=1000):
    if len(X) > sample_size:
        X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)
    relevance = mutual_info_classif(X, y)
    features = X.columns
    selected_features = []
    redundancy = np.zeros(len(features))

    for _ in range(n_features):
        mrmr_values = relevance - redundancy
        best_feature = features[np.argmax(mrmr_values)]
        selected_features.append(best_feature)

        if len(selected_features) < len(features):
            redundancies = Parallel(n_jobs=-1)(
                delayed(compute_redundancy)(feature, best_feature, X) for feature in features)
            redundancy += np.array(redundancies)

    return selected_features
# Select 20, 50, and 100 features using mRMR
selected_pixel_features_20 = mrmr_feature_selection(X_pixel_train, y_pixel_train, n_features=20)
selected_pixel_features_50 = mrmr_feature_selection(X_pixel_train, y_pixel_train, n_features=50)
selected_pixel_features_100 = mrmr_feature_selection(X_pixel_train, y_pixel_train, n_features=100)

selected_hog_features_20 = mrmr_feature_selection(X_hog_train, y_hog_train, n_features=20)
selected_hog_features_50 = mrmr_feature_selection(X_hog_train, y_hog_train, n_features=50)
selected_hog_features_100 = mrmr_feature_selection(X_hog_train, y_hog_train, n_features=100)

# Select features from training and testing sets
X_pixel_train_20 = X_pixel_train[selected_pixel_features_20]
X_pixel_test_20 = X_pixel_test[selected_pixel_features_20]
X_pixel_train_50 = X_pixel_train[selected_pixel_features_50]
X_pixel_test_50 = X_pixel_test[selected_pixel_features_50]
X_pixel_train_100 = X_pixel_train[selected_pixel_features_100]
X_pixel_test_100 = X_pixel_test[selected_pixel_features_100]

X_hog_train_20 = X_hog_train[selected_hog_features_20]
X_hog_test_20 = X_hog_test[selected_hog_features_20]
X_hog_train_50 = X_hog_train[selected_hog_features_50]
X_hog_test_50 = X_hog_test[selected_hog_features_50]
X_hog_train_100 = X_hog_train[selected_hog_features_100]
X_hog_test_100 = X_hog_test[selected_hog_features_100]

# Define the models
knn_model = KNeighborsClassifier(n_neighbors=5)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
nb_model = GaussianNB()

# Function to train and evaluate the models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    return accuracy, f1, precision, recall, report

# Evaluate models with different number of features
results = {}

# Pixel features
results['Pixel_20_kNN'] = train_and_evaluate_model(knn_model, X_pixel_train_20, y_pixel_train, X_pixel_test_20, y_pixel_test)
results['Pixel_50_kNN'] = train_and_evaluate_model(knn_model, X_pixel_train_50, y_pixel_train, X_pixel_test_50, y_pixel_test)
results['Pixel_100_kNN'] = train_and_evaluate_model(knn_model, X_pixel_train_100, y_pixel_train, X_pixel_test_100, y_pixel_test)

results['Pixel_20_RF'] = train_and_evaluate_model(rf_model, X_pixel_train_20, y_pixel_train, X_pixel_test_20, y_pixel_test)
results['Pixel_50_RF'] = train_and_evaluate_model(rf_model, X_pixel_train_50, y_pixel_train, X_pixel_test_50, y_pixel_test)
results['Pixel_100_RF'] = train_and_evaluate_model(rf_model, X_pixel_train_100, y_pixel_train, X_pixel_test_100, y_pixel_test)

results['Pixel_20_NB'] = train_and_evaluate_model(nb_model, X_pixel_train_20, y_pixel_train, X_pixel_test_20, y_pixel_test)
results['Pixel_50_NB'] = train_and_evaluate_model(nb_model, X_pixel_train_50, y_pixel_train, X_pixel_test_50, y_pixel_test)
results['Pixel_100_NB'] = train_and_evaluate_model(nb_model, X_pixel_train_100, y_pixel_train, X_pixel_test_100, y_pixel_test)

# HOG features
results['HOG_20_kNN'] = train_and_evaluate_model(knn_model, X_hog_train_20, y_hog_train, X_hog_test_20, y_hog_test)
results['HOG_50_kNN'] = train_and_evaluate_model(knn_model, X_hog_train_50, y_hog_train, X_hog_test_50, y_hog_test)
results['HOG_100_kNN'] = train_and_evaluate_model(knn_model, X_hog_train_100, y_hog_train, X_hog_test_100, y_hog_test)

results['HOG_20_RF'] = train_and_evaluate_model(rf_model, X_hog_train_20, y_hog_train, X_hog_test_20, y_hog_test)
results['HOG_50_RF'] = train_and_evaluate_model(rf_model, X_hog_train_50, y_hog_train, X_hog_test_50, y_hog_test)
results['HOG_100_RF'] = train_and_evaluate_model(rf_model, X_hog_train_100, y_hog_train, X_hog_test_100, y_hog_test)

results['HOG_20_NB'] = train_and_evaluate_model(nb_model, X_hog_train_20, y_hog_train, X_hog_test_20, y_hog_test)
results['HOG_50_NB'] = train_and_evaluate_model(nb_model, X_hog_train_50, y_hog_train, X_hog_test_50, y_hog_test)
results['HOG_100_NB'] = train_and_evaluate_model(nb_model, X_hog_train_100, y_hog_train, X_hog_test_100, y_hog_test)

# Print the results
for key, result in results.items():
    print(f"\nTest Results with {key.split('_')[1]} mRMR features for {key.split('_')[2]} ({key.split('_')[0]}):")
    print(f"Accuracy: {result[0]}, F1 Score: {result[1]}, Precision: {result[2]}, Recall: {result[3]}")
    print(result[4])
