import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the reduced CSV file
csv_file_path_reduced_sb = 'images_and_labels_reduced_SelectKBest.csv'
df_reduced_sb = pd.read_csv(csv_file_path_reduced_sb)

# Separate features and labels for training
X = df_reduced_sb.iloc[:, :-1]
y = df_reduced_sb.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifiers
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a function to save results to a text file
def save_results_to_file(filename, model_name, accuracy, report):
    with open(filename, 'a') as f:
        f.write(f"{model_name} Classifier\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write("\n" + "="*80 + "\n\n")

# 1.  KNN model
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)
print("KNN Classifier Accuracy:", accuracy_knn)
save_results_to_file('model_results_plain_pixel_sb.txt', 'KNN', accuracy_knn, report_knn)

# 2. Naive Bayes model
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Classifier Accuracy:", accuracy_nb)
report_nb = classification_report(y_test, y_pred_nb)
save_results_to_file('model_results_plain_pixel_sb.txt', 'Naive Bayes', accuracy_nb, report_nb)

# 3.  Random Forest model
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)
report_rf = classification_report(y_test, y_pred_rf)
save_results_to_file('model_results_plain_pixel_sb.txt', 'Random Forest', accuracy_rf, report_rf)
