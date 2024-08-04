import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the LDA-reduced features and labels
lda_train_df = pd.read_csv('train_lda_features.csv')
lda_test_df = pd.read_csv('test_lda_features.csv')

# Separate features and labels
X_train = lda_train_df.drop('label', axis=1)
y_train = lda_train_df['label']

X_test = lda_test_df.drop('label', axis=1)
y_test = lda_test_df['label']

# Train the Random Forest classifier to evaluate reduced features with labels
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Test Set Accuracy: {accuracy}")
print("Classification Report on Test Set:")
print(report)

# Evaluate using cross-validation
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)  # 5-fold cross-validation
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")

# Get feature importances with random classifer
feature_importances = rf_classifier.feature_importances_
indices = np.argsort(feature_importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. LDA Component {indices[f] + 1} ({feature_importances[indices[f]]})")
