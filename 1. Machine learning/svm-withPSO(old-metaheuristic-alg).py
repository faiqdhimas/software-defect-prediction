pip install pyswarm #mandatory

from google.colab import files
uploaded = files.upload()

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from pyswarm import pso

# Read dataset
data = pd.read_csv('jira305.csv')

# Handle missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown', inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Convert text features to binary indicators
data['Description'] = data['Description'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)
data['Custom field (Step to Reproduce)'] = data['Custom field (Step to Reproduce)'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)
data['Comment'] = data['Comment'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)

# Encode categorical features
categorical_columns = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Convert date columns to numeric
date_columns = ['Created', 'Updated', 'Resolved']
for col in date_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')
    data[col].fillna(data[col].median(), inplace=True)
    data[col] = data[col].astype(int) / 10**9

# Convert 'Issue Type' to binary: 1 for 'Bug', 0 for 'Not Bug'
data['Issue Type'] = data['Issue Type'].apply(lambda x: 1 if x == 'Bug' else 0)

# Select features and target
selected_features = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Created', 'Updated', 'Resolved', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']
X = data[selected_features].values
y = data['Issue Type'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PSO for feature selection
def objective_function(selected_features):
    selected_features = np.where(selected_features > 0.5, 1, 0).astype(bool)
    if np.sum(selected_features) == 0:
        return 1  # Prevent selection of no features
    X_subset = X_scaled[:, selected_features]
    model = SVC(kernel='linear', probability=True, random_state=42)

    # Using fewer folds for faster computation
    scores = cross_val_score(model, X_subset, y, cv=3, scoring='accuracy', n_jobs=-1)
    score = np.mean(scores)
    progress = 100 * (scores.size / 3)  # Calculate percentage progress
    print(f"Progress: {progress:.2f}%")

    return 1 - score  # Minimize 1 - accuracy

# Initial swarm
lb = [0] * X.shape[1]
ub = [1] * X.shape[1]

# Print features before PSO
print("Fitur sebelum PSO:")
print(selected_features)
print("Total fitur sebelum PSO:", len(selected_features))

# Apply PSO with logging
def pso_with_logging(func, lb, ub, swarmsize=20, maxiter=10):
    best_cost = float('inf')
    best_position = None
    for i in range(maxiter):
        position, cost = pso(func, lb, ub, swarmsize=swarmsize, maxiter=1)
        if cost < best_cost:
            best_cost = cost
            best_position = position
        print(f"Iteration {i + 1}/{maxiter}, Best score: {1 - best_cost:.4f}")
    return best_position, best_cost

best_features, best_score = pso_with_logging(objective_function, lb, ub, swarmsize=20, maxiter=10)

# Select features based on PSO
selected_features = np.where(best_features > 0.5, 1, 0).astype(bool)

# Print features after PSO
selected_feature_names = [name for name, selected in zip(selected_features, selected_features) if selected]
print("\nFitur setelah PSO:")
print(selected_feature_names)
print("Total fitur setelah PSO:", len(selected_feature_names))

# Split data with selected features
X_train, X_test, y_train, y_test = train_test_split(X_scaled[:, selected_features], y, test_size=0.2, random_state=42)

# Train SVM model with selected features
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])
cm = confusion_matrix(y_test, y_pred)

print(f'\nAkurasi Model: {accuracy:.2f}')
print('Classification Report:')
print(report)
print(f'AUC-ROC: {auc:.2f}')
print('Confusion Matrix:')
print(cm)

# Save model and preprocessing tools
joblib.dump(svm, 'svm_model2.pkl')
joblib.dump(scaler, 'scaler.pkl')
for col, le in label_encoders.items():
    joblib.dump(le, f'label_encoder_{col}.pkl')
