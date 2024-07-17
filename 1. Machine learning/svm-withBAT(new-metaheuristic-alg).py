from google.colab import files
uploaded = files.upload()

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

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
X = data[selected_features]
y = data['Issue Type']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print feature information before Bat Algorithm
print(f"Total features before Bat Algorithm: {X_scaled.shape[1]}")
print("Features before Bat Algorithm:", selected_features)

# Define the Bat Algorithm
class BatAlgorithm:
    def __init__(self, n_bats, n_features, n_iterations, alpha, gamma):
        self.n_bats = n_bats
        self.n_features = n_features
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.gamma = gamma
        self.population = np.random.randint(2, size=(n_bats, n_features))
        self.velocities = np.random.rand(n_bats, n_features)
        self.loudness = np.ones(n_bats)
        self.pulse_rate = np.zeros(n_bats)
        self.best_bat = self.population[0]
        self.best_score = -np.inf

    def evaluate(self, X, y, subset):
        if np.sum(subset) == 0:
            return 0
        X_subset = X[:, subset == 1]
        model = SVC(kernel='linear', probability=True, random_state=42)
        score = np.mean(cross_val_score(model, X_subset, y, cv=5, scoring='accuracy'))
        return score

    def optimize(self, X, y):
        for t in range(self.n_iterations):
            for i in range(self.n_bats):
                freq = np.random.rand()
                self.velocities[i] += (self.population[i] - self.best_bat) * freq
                candidate = self.population[i] + self.velocities[i]
                candidate = np.where(np.random.rand(self.n_features) < self.pulse_rate[i], candidate, self.population[i])
                candidate = np.clip(candidate, 0, 1).astype(int)

                if np.random.rand() < self.loudness[i]:
                    score = self.evaluate(X, y, candidate)
                    if score > self.best_score:
                        self.best_bat = candidate
                        self.best_score = score

                self.population[i] = candidate
                self.loudness[i] *= self.alpha
                self.pulse_rate[i] = self.pulse_rate[i] * (1 - np.exp(-self.gamma * t))

        return self.best_bat

# Parameters for Bat Algorithm
n_bats = 20
n_iterations = 50
alpha = 0.9
gamma = 0.9

# Run Bat Algorithm for feature selection
bat_algo = BatAlgorithm(n_bats=n_bats, n_features=X_scaled.shape[1], n_iterations=n_iterations, alpha=alpha, gamma=gamma)
best_features = bat_algo.optimize(X_scaled, y)

# Filter the selected features
X_selected = X_scaled[:, best_features == 1]

# Print feature information after Bat Algorithm
selected_feature_names = [feature for feature, selected in zip(selected_features, best_features) if selected == 1]
removed_feature_names = [feature for feature, selected in zip(selected_features, best_features) if selected == 0]

print(f"Total features after Bat Algorithm: {X_selected.shape[1]}")
print("Selected features:", selected_feature_names)
print("Removed features:", removed_feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])
cm = confusion_matrix(y_test, y_pred)

print(f'Akurasi Model: {accuracy:.2f}')
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
