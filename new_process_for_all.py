import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from joblib import Parallel, delayed

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

# Split the data into training (80%) and testing (20%)
X_pixel_train, X_pixel_test, y_pixel_train, y_pixel_test = train_test_split(X_pixel, y_pixel_encoded, test_size=0.2, random_state=42)
X_hog_train, X_hog_test, y_hog_train, y_hog_test = train_test_split(X_hog, y_hog_encoded, test_size=0.2, random_state=42)

# mRMR Feature Selection
def mrmr_feature_selection(X, y, n_features):
    relevance = mutual_info_classif(X, y)
    features = X.columns
    selected_features = []
    redundancy = np.zeros(len(features))

    def compute_redundancy(feature, best_feature, X):
        return mutual_info_regression(X[[feature]], X[best_feature])[0]

    for _ in range(n_features):
        mrmr_values = relevance - redundancy
        best_feature = features[np.argmax(mrmr_values)]
        selected_features.append(best_feature)

        if len(selected_features) < len(features):
            redundancies = Parallel(n_jobs=-1)(delayed(compute_redundancy)(feature, best_feature, X) for feature in features)
            redundancy += np.array(redundancies)

    return selected_features

# Apply mRMR on the training set
selected_pixel_features_mrmr = mrmr_feature_selection(X_pixel_train, y_pixel_train, n_features=50)
selected_hog_features_mrmr = mrmr_feature_selection(X_hog_train, y_hog_train, n_features=50)

# Select features from training and test sets
X_pixel_train_mrmr = X_pixel_train[selected_pixel_features_mrmr]
X_pixel_test_mrmr = X_pixel_test[selected_pixel_features_mrmr]

X_hog_train_mrmr = X_hog_train[selected_hog_features_mrmr]
X_hog_test_mrmr = X_hog_test[selected_hog_features_mrmr]

# Define the models for hyperparameter tuning
knn_model = KNeighborsClassifier()
nb_model_pixel = GaussianNB()
nb_model_hog = GaussianNB()
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grids
knn_param_grid = {'n_neighbors': np.arange(1, 21)}
rf_param_grid = {'n_estimators': [10, 50, 100], 'max_features': ['sqrt', 'log2']}

# Perform GridSearchCV for hyperparameter tuning
def tune_hyperparameters(model, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_estimator_

# Tune hyperparameters for both mRMR selected features
best_knn_params_pixel, best_knn_model_pixel = tune_hyperparameters(knn_model, knn_param_grid, X_pixel_train_mrmr, y_pixel_train)
best_rf_params_pixel, best_rf_model_pixel = tune_hyperparameters(rf_model, rf_param_grid, X_pixel_train_mrmr, y_pixel_train)

best_knn_params_hog, best_knn_model_hog = tune_hyperparameters(knn_model, knn_param_grid, X_hog_train_mrmr, y_hog_train)
best_rf_params_hog, best_rf_model_hog = tune_hyperparameters(rf_model, rf_param_grid, X_hog_train_mrmr, y_hog_train)

# Validate the models on the training set using cross-validation
def cross_val_model(model, X_train, y_train, cv=5):
    accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_weighted')
    recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall_weighted')
    return accuracy.mean(), f1.mean(), precision.mean(), recall.mean()

# Train Naive Bayes model on the training set
nb_model_pixel.fit(X_pixel_train_mrmr, y_pixel_train)
nb_model_hog.fit(X_hog_train_mrmr, y_hog_train)

# Cross-validate models with mRMR features
knn_cv_accuracy_pixel, knn_cv_f1_pixel, knn_cv_precision_pixel, knn_cv_recall_pixel = cross_val_model(best_knn_model_pixel, X_pixel_train_mrmr, y_pixel_train)
rf_cv_accuracy_pixel, rf_cv_f1_pixel, rf_cv_precision_pixel, rf_cv_recall_pixel = cross_val_model(best_rf_model_pixel, X_pixel_train_mrmr, y_pixel_train)
nb_cv_accuracy_pixel, nb_cv_f1_pixel, nb_cv_precision_pixel, nb_cv_recall_pixel = cross_val_model(nb_model_pixel, X_pixel_train_mrmr, y_pixel_train)

knn_cv_accuracy_hog, knn_cv_f1_hog, knn_cv_precision_hog, knn_cv_recall_hog = cross_val_model(best_knn_model_hog, X_hog_train_mrmr, y_hog_train)
rf_cv_accuracy_hog, rf_cv_f1_hog, rf_cv_precision_hog, rf_cv_recall_hog = cross_val_model(best_rf_model_hog, X_hog_train_mrmr, y_hog_train)
nb_cv_accuracy_hog, nb_cv_f1_hog, nb_cv_precision_hog, nb_cv_recall_hog = cross_val_model(nb_model_hog, X_hog_train_mrmr, y_hog_train)

# Evaluate the models on the test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    return accuracy, f1, precision, recall, report

# Evaluate models with mRMR features
knn_test_accuracy_pixel, knn_test_f1_pixel, knn_test_precision_pixel, knn_test_recall_pixel, knn_test_report_pixel = evaluate_model(best_knn_model_pixel, X_pixel_test_mrmr, y_pixel_test)
rf_test_accuracy_pixel, rf_test_f1_pixel, rf_test_precision_pixel, rf_test_recall_pixel, rf_test_report_pixel = evaluate_model(best_rf_model_pixel, X_pixel_test_mrmr, y_pixel_test)
nb_test_accuracy_pixel, nb_test_f1_pixel, nb_test_precision_pixel, nb_test_recall_pixel, nb_test_report_pixel = evaluate_model(nb_model_pixel, X_pixel_test_mrmr, y_pixel_test)

knn_test_accuracy_hog, knn_test_f1_hog, knn_test_precision_hog, knn_test_recall_hog, knn_test_report_hog = evaluate_model(best_knn_model_hog, X_hog_test_mrmr, y_hog_test)
rf_test_accuracy_hog, rf_test_f1_hog, rf_test_precision_hog, rf_test_recall_hog, rf_test_report_hog = evaluate_model(best_rf_model_hog, X_hog_test_mrmr, y_hog_test)
nb_test_accuracy_hog, nb_test_f1_hog, nb_test_precision_hog, nb_test_recall_hog, nb_test_report_hog = evaluate_model(nb_model_hog, X_hog_test_mrmr, y_hog_test)

# Print the results
print("\nCross-Validation Results with mRMR features for kNN (Pixel):")
print(f"Accuracy: {knn_cv_accuracy_pixel}, F1 Score: {knn_cv_f1_pixel}, Precision: {knn_cv_precision_pixel}, Recall: {knn_cv_recall_pixel}")
print("\nCross-Validation Results with mRMR features for Random Forest (Pixel):")
print(f"Accuracy: {rf_cv_accuracy_pixel}, F1 Score: {rf_cv_f1_pixel}, Precision: {rf_cv_precision_pixel}, Recall: {rf_cv_recall_pixel}")
print("\nCross-Validation Results with mRMR features for Naive Bayes (Pixel):")
print(f"Accuracy: {nb_cv_accuracy_pixel}, F1 Score: {nb_cv_f1_pixel}, Precision: {nb_cv_precision_pixel}, Recall: {nb_cv_recall_pixel}")

print("\nCross-Validation Results with mRMR features for kNN (HOG):")
print(f"Accuracy: {knn_cv_accuracy_hog}, F1 Score: {knn_cv_f1_hog}, Precision: {knn_cv_precision_hog}, Recall: {knn_cv_recall_hog}")
print("\nCross-Validation Results with mRMR features for Random Forest (HOG):")
print(f"Accuracy: {rf_cv_accuracy_hog}, F1 Score: {rf_cv_f1_hog}, Precision: {rf_cv_precision_hog}, Recall: {rf_cv_recall_hog}")
print("\nCross-Validation Results with mRMR features for Naive Bayes (HOG):")
print(f"Accuracy: {nb_cv_accuracy_hog}, F1 Score: {nb_cv_f1_hog}, Precision: {nb_cv_precision_hog}, Recall: {nb_cv_recall_hog}")

print("\nTest Results with mRMR features for kNN (Pixel):")
print(knn_test_report_pixel)
print("\nTest Results with mRMR features for Random Forest (Pixel):")
print(rf_test_report_pixel)
print("\nTest Results with mRMR features for Naive Bayes (Pixel):")
print(nb_test_report_pixel)

print("\nTest Results with mRMR features for kNN (HOG):")
print(knn_test_report_hog)
print("\nTest Results with mRMR features for Random Forest (HOG):")
print(rf_test_report_hog)
print("\nTest Results with mRMR features for Naive Bayes (HOG):")
print(nb_test_report_hog)
