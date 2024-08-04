import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Load the dataset from csv file
data_file_name = 'gesture_language_digits_hog_features.csv'
data = pd.read_csv(data_file_name)

# Split features and labels
X = data.drop(columns=['label']).values
y = data['label'].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply LDA with number of classes-1 features
lda = LDA(n_components=9)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Convert the reduced features to DataFrame
train_lda_df = pd.DataFrame(X_train_lda, columns=[f'lda_component_{i+1}' for i in range(X_train_lda.shape[1])])
train_lda_df['label'] = y_train

test_lda_df = pd.DataFrame(X_test_lda, columns=[f'lda_component_{i+1}' for i in range(X_test_lda.shape[1])])
test_lda_df['label'] = y_test

# Save to CSV
train_lda_df.to_csv('train_lda_reduced_features_hog.csv', index=False)
test_lda_df.to_csv('test_lda_reduced_features_hog.csv', index=False)
print("LDA reduced features dataset created successfully.")

def save_results_to_file(filename, model_name, accuracy, report):
    with open(filename, 'a') as f:
        f.write(f"{model_name} Classifier\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write("\n" + "="*80 + "\n\n")


# 1. KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_lda, y_train)
knn_predictions = knn.predict(X_test_lda)
accuracy_knn = accuracy_score(y_test, knn_predictions)
report_knn = classification_report(y_test, knn_predictions)
print("KNN Accuracy:", accuracy_score(y_test, knn_predictions))
save_results_to_file('model_results_hog_lda.txt', 'KNN', accuracy_knn, report_knn)

# 2. Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train_lda, y_train)
nb_predictions = nb.predict(X_test_lda)
accuracy_nb = accuracy_score(y_test, nb_predictions)
report_nb = classification_report(y_test, nb_predictions)
save_results_to_file('model_results_hog_lda.txt', 'Naive Bayes', accuracy_nb, report_nb)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))

# 3. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_lda, y_train)
rf_predictions = rf.predict(X_test_lda)
accuracy_rf = accuracy_score(y_test, rf_predictions)
print("Random Forest Classifier Accuracy:", accuracy_rf)
report_rf = classification_report(y_test, rf_predictions)
save_results_to_file('model_results_hog_lda.txt', 'Random Forest', accuracy_rf, report_rf)

