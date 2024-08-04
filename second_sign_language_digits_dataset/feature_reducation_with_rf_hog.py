# Identify important features using RFE for Random Forest

# # Recursive Feature Elimination with Random Forest
# rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=10)
# rfe.fit(X, y)
#
# # Get the ranking of features
# rfe_ranking = rfe.ranking_
#
# # Display the features ranking
# important_features = np.argsort(rfe_ranking)[:10]
# print("Top 10 Important Features (0-based indices):", important_features)
#
# #  Use feature importances directly from the Random Forest model
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# feature_importances = rf.feature_importances_
# important_features_rf = np.argsort(feature_importances)[-10:]
# print("Top 10 Important Features from Random Forest (0-based indices):", important_features_rf)
# # Train and evaluate models