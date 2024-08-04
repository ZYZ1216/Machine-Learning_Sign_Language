import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_curve, \
    auc

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
X_pixel_train, X_pixel_test, y_pixel_train, y_pixel_test = train_test_split(X_pixel, y_pixel_encoded, test_size=0.2,
                                                                            random_state=42)
X_hog_train, X_hog_test, y_hog_train, y_hog_test = train_test_split(X_hog, y_hog_encoded, test_size=0.2,
                                                                    random_state=42)

# Define the models for feature selection and evaluation
rf_model = RandomForestClassifier(random_state=42)
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()

# Define the hyperparameter grids
knn_param_grid = {'n_neighbors': np.arange(1, 22)}
rf_param_grid = {'n_estimators': np.arange(10, 110, 10), 'max_features': ['sqrt', 'log2']}


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

# Perform GridSearchCV for hyperparameter tuning
def tune_hyperparameters(model, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_estimator_


# Evaluate the models on the test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    return accuracy, f1, precision, recall, report


# Generate ROC curves for the models
def plot_roc_curves(results, model_name):
    plt.figure(figsize=(10, 8))
    for res in results:
        if res['fpr'] is not None and res['tpr'] is not None:
            plt.plot(res['fpr'], res['tpr'],
                     label=f"{model_name} with {res['n_features']} features (AUC = {res['roc_auc']:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()


# Select different numbers of features using mRMR
def select_features_and_evaluate(X_train, y_train, X_test, y_test, n_features_list, model, param_grid, model_name):
    results = []
    for n_features in n_features_list:
        selected_features = mrmr_feature_selection(X_train, y_train, n_features=n_features)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        best_params, best_model = tune_hyperparameters(model, param_grid, X_train_selected, y_train)
        test_accuracy, test_f1, test_precision, test_recall, test_report = evaluate_model(best_model, X_test_selected,
                                                                                          y_test)

        if hasattr(best_model, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_selected)[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None

        results.append({
            'n_features': n_features,
            'best_params': best_params,
            'accuracy': test_accuracy,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'report': test_report
        })

        print(f"\nTest Results with {n_features} mRMR features for {model_name}:")
        print(test_report)

    return results


# Define the numbers of features to select
n_features_list = [20, 50, 100]


# Evaluate kNN for different values of k
def evaluate_knn(X_train, y_train, X_test, y_test, n_features_list):
    results = []
    for n_features in n_features_list:
        selected_features = mrmr_feature_selection(X_train, y_train, n_features=n_features)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        accuracies = []
        for k in range(1, 22):
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train_selected, y_train)
            test_accuracy = knn_model.score(X_test_selected, y_test)
            accuracies.append(test_accuracy)
            print(f"kNN with {n_features} features and k={k}: Accuracy = {test_accuracy:.4f}")

        results.append({
            'n_features': n_features,
            'accuracies': accuracies
        })

    return results


# Evaluate Random Forest for different numbers of estimators
def evaluate_rf(X_train, y_train, X_test, y_test, n_features_list):
    results = []
    for n_features in n_features_list:
        selected_features = mrmr_feature_selection(X_train, y_train, n_features=n_features)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        accuracies = []
        for n_estimators in range(10, 110, 10):
            rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            rf_model.fit(X_train_selected, y_train)
            test_accuracy = rf_model.score(X_test_selected, y_test)
            accuracies.append(test_accuracy)
            print(
                f"Random Forest with {n_features} features and n_estimators={n_estimators}: Accuracy = {test_accuracy:.4f}")

        results.append({
            'n_features': n_features,
            'accuracies': accuracies
        })

    return results


# Plot the performance of kNN for different values of k
def plot_knn_performance(knn_results):
    plt.figure(figsize=(10, 8))
    for res in knn_results:
        plt.plot(range(1, 22), res['accuracies'], label=f"{res['n_features']} features")
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('kNN Performance for Different Numbers of Features')
    plt.legend()
    plt.show()


# Plot the performance of Random Forest for different numbers of estimators
def plot_rf_performance(rf_results):
    plt.figure(figsize=(10, 8))
    for res in rf_results:
        plt.plot(range(10, 110, 10), res['accuracies'], label=f"{res['n_features']} features")
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Performance for Different Numbers of Features')
    plt.legend()
    plt.show()


# Perform evaluation
knn_results = evaluate_knn(X_pixel_train, y_pixel_train, X_pixel_test, y_pixel_test, n_features_list)
rf_results = evaluate_rf(X_pixel_train, y_pixel_train, X_pixel_test, y_pixel_test, n_features_list)

# Plot the results
plot_knn_performance(knn_results)
plot_rf_performance(rf_results)
