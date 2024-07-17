from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# Membaca dataset
# data = pd.read_csv('/mnt/data/jun9-traindataset195an.csv')
data = pd.read_csv('jira305.csv')

# Mengisi missing values dengan metode yang sesuai
for col in data.columns:
    if data[col].dtype == 'object':
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].dropna().mode().empty else 'unknown')
    else:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())

# Convert text features to binary indicators
data['Description'] = data['Description'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)
data['Custom field (Step to Reproduce)'] = data['Custom field (Step to Reproduce)'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)
data['Comment'] = data['Comment'].apply(lambda x: 1 if pd.notna(x) and x.strip() else 0)

# Encoding categorical features
label_encoders = {}
categorical_columns = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Pilih fitur yang relevan untuk klasifikasi
selected_features = ['Status', 'Project key', 'Project name', 'Project type', 'Project lead', 'Priority', 'Assignee', 'Reporter', 'Creator', 'Created', 'Updated', 'Resolved', 'Components', 'Labels', 'Custom field (Story Points)', 'Custom field (TE Assignee)', 'Parent', 'Parent summary', 'Description', 'Custom field (Step to Reproduce)', 'Watchers', 'Custom field (Apps Name)', 'Custom field (Assignee Squad)', 'Comment']

# Convert date columns to datetime
data['Created'] = pd.to_datetime(data['Created'], errors='coerce')
data['Updated'] = pd.to_datetime(data['Updated'], errors='coerce')
data['Resolved'] = pd.to_datetime(data['Resolved'], errors='coerce')

# Replace NaT with median time
data['Created'].fillna(data['Created'].median(), inplace=True)
data['Updated'].fillna(data['Updated'].median(), inplace=True)
data['Resolved'].fillna(data['Resolved'].median(), inplace=True)

# Convert date features to numeric (seconds since epoch)
data['Created'] = data['Created'].astype(int) / 10**9
data['Updated'] = data['Updated'].astype(int) / 10**9
data['Resolved'] = data['Resolved'].astype(int) / 10**9

# Features and target label
X = data[selected_features]
y = data['Issue Type'].apply(lambda x: 1 if x == 'Bug' else 0)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Melatih model Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred = rf_model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
print(f'AUC-ROC: {roc_auc:.2f}')
print('Confusion Matrix:')
print(cm)

# Save the model and encoders
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

for col, le in label_encoders.items():
    joblib.dump(le, f'label_encoder_{col}.pkl')
