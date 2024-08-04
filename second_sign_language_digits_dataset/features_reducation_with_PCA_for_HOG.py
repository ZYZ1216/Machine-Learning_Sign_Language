import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#  import the CSV file with HOG features
csv_filename = 'gesture_language_digits_hog_features.csv'
data = pd.read_csv(csv_filename)

# Split features and labels
X = data.drop(columns=['label']).values
y = data['label'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reduce the number of features using PCA to 50
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Combine reduced features with labels
data_reduced_pca = np.column_stack((X_pca, y))
df_reduced_pca = pd.DataFrame(data_reduced_pca)
# Save to a new CSV file for reduced features with PCA
csv_file_path_reduced_pca = 'gesture_language_digits_hog_pca.csv'
df_reduced_pca.to_csv(csv_file_path_reduced_pca, index=False)
print(f'Reduced data saved to {csv_file_path_reduced_pca}')

def save_results_to_file(filename, model_name, accuracy, report):
    with open(filename, 'a') as f:
        f.write(f"{model_name} Classifier\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write("\n" + "="*80 + "\n\n")

# Split the data into training and testing sets for model training and testing
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 1. KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, knn_predictions)
report_knn = classification_report(y_test, knn_predictions)
print("KNN Accuracy:", accuracy_score(y_test, knn_predictions))
save_results_to_file('model_results_hog_pca.txt', 'KNN', accuracy_knn, report_knn)

# 2. Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, nb_predictions)
report_nb = classification_report(y_test, nb_predictions)
save_results_to_file('model_results_hog_pca.txt', 'Naive Bayes', accuracy_nb, report_nb)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))

# 3. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, rf_predictions)
print("Random Forest Classifier Accuracy:", accuracy_rf)
report_rf = classification_report(y_test, rf_predictions)
save_results_to_file('model_results_hog_pca.txt', 'Random Forest', accuracy_rf, report_rf)





